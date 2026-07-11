"""
Genetic Hamiltonian Population — competing field variants.

Runs N copies of the quantum field, each with a different candidate
Hamiltonian. Selection pressure from prediction accuracy converges
toward the true dynamics of the world.

Each individual owns:
    - A HamiltonianSpec per cluster (coefficient vector)
    - A QuantumField (density matrices evolving under its H)
    - A fitness score (EMA of prediction accuracy)

The population lifecycle:
    1. All individuals evolve in parallel on each sensor event
    2. Prediction residuals update fitness scores
    3. Every generation_interval steps: tournament selection,
       crossover, mutation, elite preservation
    4. Best individual's H is periodically injected into the
       production field
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field

import numpy as np
from numpy.typing import NDArray

from umwelt.substrate.field import QuantumField
from umwelt.substrate.hamiltonian import HamiltonianSpec
from umwelt.substrate.graph import WorldGraph

logger = logging.getLogger(__name__)

# The genetic population gives each individual a DENSE HamiltonianSpec per cluster
# (the 2^n operator table summed in build()). That's the crossover/mutation substrate
# — but a big cumulant cluster (the origin's 26q region-merge manifold, 15q entities) has no
# affordable dense basis: a 26q operator table OOMs the box at construction (5.5GB+).
# Those big clusters are CUMULANT and learn through the fractal tower + readout +
# observe-collapse; the genetic H-search is just one learner among several, and a
# genetic search over a 26q Hamiltonian is dubious value anyway. So the population
# SKIPS clusters above this size — they keep every other learner, just not the genetic
# one. Current max live cluster is 7q (exterior), so this skips nothing today; it's the
# guard that lets the region/person merges build. See project_construction_oom.
_POP_MAX_QUBITS = 10


@dataclass
class PopulationConfig:
    """Configuration for genetic Hamiltonian search."""

    size: int = 8
    elite_count: int = 2
    mutation_sigma: float = 0.05
    crossover_rate: float = 0.3
    tournament_size: int = 3
    generation_interval: int = 100
    min_age: int = 50
    fitness_ema_alpha: float = 0.05
    inject_interval: int = 100
    initial_perturbation: float = 0.01
    enabled: bool = False


@dataclass
class Individual:
    """One competing Hamiltonian candidate."""

    id: int
    h_specs: dict[str, HamiltonianSpec]
    field: QuantumField
    fitness: float = 0.5
    age: int = 0
    parents: list[int] = dc_field(default_factory=list)   # lineage: parent ids (empty=seed)
    generation_born: int = 0
    _prev_z: dict[str, dict[str, float]] = dc_field(default_factory=dict)


class Population:
    """
    Genetic population of competing Hamiltonian candidates.

    Each individual has its own QuantumField and HamiltonianSpec
    per cluster. They share the WorldGraph topology (read-only).
    """

    def __init__(
        self,
        graph: WorldGraph,
        config: PopulationConfig,
        gamma: float = 0.05,
        dt: float = 0.01,
        bridge_strength: float = 0.5,
        seed: int | None = None,
    ):
        from umwelt.substrate.params import ParameterBundle

        self.graph = graph
        self.config = config
        self._gamma = gamma
        self._dt = dt
        self._bridge_strength = bridge_strength
        self._step = 0
        self._generation = 0
        self._rng = np.random.default_rng(seed)
        # Inner-Population lineage log (additive; ON by default, env-disableable). Each
        # generation appends one row per individual to a JSONL the shelf reader traverses.
        from pathlib import Path as _Path
        from umwelt._util import env_flag
        self._lineage_enabled = env_flag("UMWELT_POP_LINEAGE", default=True)
        self._lineage_path = _Path(__file__).resolve().parents[2] / "var" / "shelf" / "population_lineage.jsonl"

        # Learnable genetic parameters — controls exploration/exploitation.
        # mutation_sigma: how much offspring deviate from parents
        # crossover_rate: gene mixing probability
        # fitness_ema_alpha: how fast fitness tracking responds
        self.params = ParameterBundle.from_dict({
            "mutation_sigma": (config.mutation_sigma, 0.02, 0.001, 0.5),
            "crossover_rate": (config.crossover_rate, 0.1, 0.05, 0.95),
            "fitness_ema_alpha": (config.fitness_ema_alpha, 0.02, 0.01, 0.3),
        })

        self.individuals: list[Individual] = []
        for i in range(config.size):
            ind = self._create_individual(i)
            self.individuals.append(ind)

        logger.info(
            "Population: %d individuals, %d H-terms per cluster",
            len(self.individuals),
            self._count_terms(),
        )

    def _create_individual(
        self,
        id: int,
        coefficients: dict[str, NDArray] | None = None,
    ) -> Individual:
        """Create a new individual with its own field and H specs."""
        field = QuantumField(
            graph=self.graph,
            gamma=self._gamma,
            dt=self._dt,
            bridge_strength=self._bridge_strength,
        )

        h_specs: dict[str, HamiltonianSpec] = {}
        for name, cluster in field.clusters.items():
            if cluster.n_qubits > _POP_MAX_QUBITS:
                continue  # too big for a dense genetic basis — see _POP_MAX_QUBITS
            spec = HamiltonianSpec(cluster.n_qubits, cluster.qubit_roles)
            if coefficients and name in coefficients:
                spec.set_coefficients(coefficients[name])
            elif self.config.initial_perturbation > 0:
                perturbation = self._rng.normal(
                    0, self.config.initial_perturbation, spec.n_terms
                )
                spec.set_coefficients(perturbation)
            h_specs[name] = spec

        # Apply initial H to the field
        field.apply_hamiltonian(h_specs)

        return Individual(id=id, h_specs=h_specs, field=field)

    def _count_terms(self) -> int:
        """Count H terms in the first individual (all have the same structure)."""
        if not self.individuals:
            return 0
        return sum(s.n_terms for s in self.individuals[0].h_specs.values())

    # ================================================================
    # Main step
    # ================================================================

    def step(
        self,
        inputs: dict[str, NDArray] | None = None,
    ):
        """
        Evolve all individuals and update fitness.

        Call this with the same inputs the production field receives.
        """
        for ind in self.individuals:
            # Record predictions (pre-step Bloch z)
            self._record_predictions(ind)

            # Evolve under this individual's H
            ind.field.step(inputs)
            ind.age += 1

            # Compute prediction residual and update fitness
            if ind.age > self.config.min_age:
                residual = self._compute_residual(ind)
                accuracy = 1.0 / (1.0 + residual)
                alpha = self.params.get("fitness_ema_alpha")
                ind.fitness = alpha * accuracy + (1.0 - alpha) * ind.fitness

        self._step += 1

        # Run selection at generation boundaries
        if self._step % self.config.generation_interval == 0:
            self._select()

    def _record_predictions(self, ind: Individual):
        """Snapshot Bloch z-values before the step (the prediction)."""
        ind._prev_z = {}
        for name, cluster in ind.field.clusters.items():
            ind._prev_z[name] = {
                role: float(cluster.qubit_bloch(idx)[2])
                for role, idx in cluster.role_index.items()
            }

    def _compute_residual(self, ind: Individual) -> float:
        """Mean squared residual between predicted and actual Bloch z."""
        total_sq = 0.0
        count = 0
        for name, cluster in ind.field.clusters.items():
            prev = ind._prev_z.get(name, {})
            for role, idx in cluster.role_index.items():
                predicted = prev.get(role, 0.0)
                actual = float(cluster.qubit_bloch(idx)[2])
                total_sq += (actual - predicted) ** 2
                count += 1
        return total_sq / max(count, 1)

    # ================================================================
    # Genetic operators
    # ================================================================

    def _select(self):
        """Tournament selection + crossover + mutation. Elites preserved."""
        # Sort by fitness (best first)
        ranked = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)

        # Elites survive unchanged
        new_gen: list[Individual] = []
        for i in range(min(self.config.elite_count, len(ranked))):
            new_gen.append(ranked[i])

        # Fill rest with offspring
        next_id = max(ind.id for ind in self.individuals) + 1
        while len(new_gen) < self.config.size:
            parent_a = self._tournament()
            parent_b = self._tournament()

            child_coeffs = {}
            for name in parent_a.h_specs:
                ca = parent_a.h_specs[name].coefficients
                cb = parent_b.h_specs[name].coefficients

                # Uniform crossover (learnable rate)
                mask = self._rng.random(len(ca)) < self.params.get("crossover_rate")
                child = np.where(mask, cb, ca)

                # Gaussian mutation (learnable sigma)
                child += self._rng.normal(0, self.params.get("mutation_sigma"), len(child))
                child = np.clip(child, -2.0, 2.0)

                child_coeffs[name] = child

            new_ind = self._create_individual(next_id, child_coeffs)
            new_ind.parents = [parent_a.id, parent_b.id]      # lineage edge
            new_ind.generation_born = self._generation + 1
            new_gen.append(new_ind)
            next_id += 1

        # Meta-learning: adapt mutation/crossover from generational fitness trend
        mean_fitness = float(np.mean([i.fitness for i in new_gen]))
        prev_mean = getattr(self, '_prev_gen_fitness', mean_fitness)
        fitness_delta = mean_fitness - prev_mean

        if self._generation > 0:
            # Use the tower's shared step factors (live-read from root bundle,
            # fallback to module defaults). Same "optimizer step" vocabulary as
            # CalibrationLoop and FractalScale — one named home, not bare literals.
            from umwelt.learning.meta_idioms import tower_steps
            root = getattr(self.graph, "root", None)
            steps = tower_steps(root.param_bundle if root is not None else None)
            current_ms = self.params.get("mutation_sigma")
            if fitness_delta > 0:
                # Improving — tighten mutation (exploit), keep crossover
                self.params.update("mutation_sigma",
                                   current_ms * steps["step_down"], 0.01)
            else:
                # Stagnating or declining — widen mutation (explore)
                self.params.update("mutation_sigma",
                                   current_ms * steps["step_up_bold"], 0.01)
                # Also increase crossover to mix genes more
                current_cr = self.params.get("crossover_rate")
                self.params.update("crossover_rate",
                                   min(0.9, current_cr * steps["step_up"]), 0.02)

        self._prev_gen_fitness = mean_fitness

        self.individuals = new_gen
        self._generation += 1
        self._log_generation()

        best = self.best
        logger.info(
            "Generation %d: best_fitness=%.4f (id=%d), mean=%.4f",
            self._generation,
            best.fitness,
            best.id,
            mean_fitness,
        )

    def _log_generation(self):
        """Append one lineage row per individual to population_lineage.jsonl (additive;
        the shelf reader traverses who-bred-whom + the fitness trajectory). Fully guarded
        — never crashes the genetic loop."""
        if not getattr(self, "_lineage_enabled", False):
            return
        try:
            import json
            self._lineage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._lineage_path, "a") as fh:
                for ind in self.individuals:
                    fh.write(json.dumps({
                        "gen": self._generation, "id": ind.id,
                        "parents": list(ind.parents),
                        "fitness": round(float(ind.fitness), 5),
                        "generation_born": ind.generation_born,
                    }) + "\n")
        except Exception:
            pass

    def _tournament(self) -> Individual:
        """Tournament selection: pick k random, return the fittest."""
        k = min(self.config.tournament_size, len(self.individuals))
        candidates = self._rng.choice(self.individuals, size=k, replace=False)
        return max(candidates, key=lambda x: x.fitness)

    # ================================================================
    # Injection + readout
    # ================================================================

    @property
    def best(self) -> Individual:
        """Current fittest individual."""
        return max(self.individuals, key=lambda x: x.fitness)

    def inject(self, target_field: QuantumField):
        """Copy best individual's H into the production field."""
        best = self.best
        target_field.apply_hamiltonian(best.h_specs)
        logger.info(
            "Injected H from individual %d (fitness=%.4f) into production field",
            best.id, best.fitness,
        )

    # ================================================================
    # Persistence
    # ================================================================

    def snapshot(self) -> dict:
        """Serializable population state."""
        return {
            "generation": self._generation,
            "step": self._step,
            "individuals": [
                {
                    "id": ind.id,
                    "fitness": ind.fitness,
                    "age": ind.age,
                    "parents": list(ind.parents),
                    "generation_born": ind.generation_born,
                    "h_specs": {
                        name: spec.snapshot()
                        for name, spec in ind.h_specs.items()
                    },
                }
                for ind in self.individuals
            ],
        }

    def load_snapshot(self, data: dict):
        """Restore population from snapshot."""
        self._generation = data.get("generation", 0)
        self._step = data.get("step", 0)

        for ind_data in data.get("individuals", []):
            coefficients = {}
            for name, spec_data in ind_data.get("h_specs", {}).items():
                spec = HamiltonianSpec.from_snapshot(spec_data)
                coefficients[name] = spec.coefficients

            ind = self._create_individual(ind_data["id"], coefficients)
            ind.fitness = ind_data.get("fitness", 0.5)
            ind.age = ind_data.get("age", 0)
            ind.parents = list(ind_data.get("parents", []))         # lineage survives resume
            ind.generation_born = ind_data.get("generation_born", 0)

            # Replace matching individual or append
            replaced = False
            for i, existing in enumerate(self.individuals):
                if existing.id == ind.id:
                    self.individuals[i] = ind
                    replaced = True
                    break
            if not replaced and len(self.individuals) < self.config.size:
                self.individuals.append(ind)

        logger.info(
            "Population restored: gen=%d, %d individuals",
            self._generation, len(self.individuals),
        )

    def stats(self) -> dict:
        """Population diagnostics."""
        fitnesses = [ind.fitness for ind in self.individuals]
        best = self.best
        return {
            "enabled": self.config.enabled,
            "generation": self._generation,
            "step": self._step,
            "population_size": len(self.individuals),
            "best_fitness": best.fitness,
            "best_id": best.id,
            "mean_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "individuals": [
                {
                    "id": ind.id,
                    "fitness": round(ind.fitness, 4),
                    "age": ind.age,
                }
                for ind in self.individuals
            ],
            "best_coefficients": {
                name: {
                    label: round(float(v), 6)
                    for label, v in zip(spec.basis.labels, spec.coefficients)
                }
                for name, spec in best.h_specs.items()
            },
        }
