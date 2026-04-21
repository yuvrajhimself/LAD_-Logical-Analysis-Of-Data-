import numpy as np


class GeneticRuleMiner:
    """
    Genetic Algorithm Rule Miner for LAD.

    Individual encoding: { attrs, values, label }
    Fitness = purity * coverage_fraction
    Fitness sharing penalises similar individuals to maintain diversity.

    New in this version
    -------------------
    - Fitness sharing via Jaccard similarity on attr sets (controlled by
      GA_SHARING_SIGMA). Set to 0.0 to disable.
    - Weight denominator uses true class counts (not accumulated rule support).
    - h_weight removed (GA does not use A* heuristic).
    """

    def __init__(self,
                 binarizer=None,
                 selector=None,
                 purity=0.6,
                 n_generations=300,
                 pop_size=200,
                 min_attrs=1,
                 max_attrs=None,
                 tournament_size=5,
                 crossover_rate=0.7,
                 mutation_rate=0.15,
                 elite_frac=0.1,
                 sharing_sigma=0.3,
                 threshold=0,
                 verbose=True,
                 random_state=42):

        self.binarizer      = binarizer
        self.selector       = selector
        self.purity         = purity
        self.n_generations  = n_generations
        self.pop_size       = pop_size
        self.min_attrs      = min_attrs
        self.max_attrs      = max_attrs
        self.tournament_size= tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate  = mutation_rate
        self.elite_frac     = elite_frac
        self.sharing_sigma  = sharing_sigma
        self.threshold      = threshold
        self.verbose        = verbose
        self.random_state   = random_state
        self.rules          = []

    # ── Individual encoding ───────────────────────────────────────────────────

    def _random_individual(self, n_features):
        max_a  = self.max_attrs or max(1, n_features // 2)
        size   = self.rng.integers(self.min_attrs, max_a + 1)
        attrs  = sorted(self.rng.choice(n_features, size=size, replace=False).tolist())
        values = self.rng.integers(0, 2, size=len(attrs)).tolist()
        label  = int(self.rng.integers(0, 2))
        return {"attrs": attrs, "values": values, "label": label}

    # ── Fitness ───────────────────────────────────────────────────────────────

    def _fitness(self, ind, Xn, y, label_counts):
        attrs, values, label = ind["attrs"], ind["values"], ind["label"]
        if not attrs:
            return 0.0
        mask    = np.all(Xn[:, attrs] == values, axis=1)
        covered = np.where(mask)[0]
        repet   = len(covered)
        if repet <= self.threshold:
            return 0.0
        correct = np.sum(y[covered] == label)
        purity  = correct / repet
        if purity < self.purity:
            return 0.0
        total_label = label_counts.get(label, 1)
        coverage    = correct / total_label
        return purity * coverage

    # ── Fitness sharing ───────────────────────────────────────────────────────

    def _apply_sharing(self, population, fitnesses):
        """
        Penalise individuals that are too similar to other high-fitness
        individuals. Similarity is Jaccard index on attribute sets.
        sigma=0.0 disables sharing entirely.
        """
        if self.sharing_sigma <= 0.0:
            return fitnesses.copy()

        shared = fitnesses.copy()
        n      = len(population)

        for i in range(n):
            if fitnesses[i] == 0:
                continue
            attrs_i     = set(population[i]["attrs"])
            niche_count = 0.0

            for j in range(n):
                if fitnesses[j] == 0:
                    continue
                attrs_j = set(population[j]["attrs"])
                union   = attrs_i | attrs_j
                if not union:
                    similarity = 1.0
                else:
                    similarity = len(attrs_i & attrs_j) / len(union)
                if similarity > self.sharing_sigma:
                    niche_count += 1.0 - similarity / self.sharing_sigma

            shared[i] = fitnesses[i] / max(1.0, niche_count)

        return shared

    # ── Selection ─────────────────────────────────────────────────────────────

    def _tournament_select(self, population, fitnesses):
        idxs = self.rng.choice(len(population),
                               size=self.tournament_size, replace=False)
        best = max(idxs, key=lambda i: fitnesses[i])
        return population[best]

    # ── Crossover ─────────────────────────────────────────────────────────────

    def _crossover(self, p1, p2, n_features):
        if self.rng.random() > self.crossover_rate:
            return dict(p1), dict(p2)

        all_attrs = sorted(set(p1["attrs"]) | set(p2["attrs"]))
        if len(all_attrs) < 2:
            return dict(p1), dict(p2)

        split  = self.rng.integers(1, len(all_attrs))
        attrs1 = all_attrs[:split]
        attrs2 = all_attrs[split:]

        def make_child(attrs, ref1, ref2, label):
            ref_vals = {}
            for a, v in zip(ref1["attrs"], ref1["values"]):
                ref_vals[a] = v
            for a, v in zip(ref2["attrs"], ref2["values"]):
                if a not in ref_vals:
                    ref_vals[a] = v
            values = [ref_vals.get(a, int(self.rng.integers(0, 2)))
                      for a in attrs]
            return {"attrs": attrs, "values": values, "label": label}

        label1 = p1["label"] if self.rng.random() < 0.5 else p2["label"]
        label2 = p2["label"] if self.rng.random() < 0.5 else p1["label"]
        c1 = make_child(attrs1 or [all_attrs[0]],  p1, p2, label1)
        c2 = make_child(attrs2 or [all_attrs[-1]], p1, p2, label2)
        return c1, c2

    # ── Mutation ──────────────────────────────────────────────────────────────

    def _mutate(self, ind, n_features):
        ind   = {"attrs": ind["attrs"][:],
                 "values": ind["values"][:],
                 "label": ind["label"]}
        max_a = self.max_attrs or max(1, n_features // 2)

        for i in range(len(ind["attrs"])):
            if self.rng.random() < self.mutation_rate:
                ind["values"][i] = 1 - ind["values"][i]

        if self.rng.random() < self.mutation_rate:
            unused = list(set(range(n_features)) - set(ind["attrs"]))
            if unused:
                swap_pos = int(self.rng.integers(0, len(ind["attrs"])))
                ind["attrs"][swap_pos] = int(self.rng.choice(unused))
                ind["attrs"] = sorted(ind["attrs"])

        if self.rng.random() < self.mutation_rate:
            if len(ind["attrs"]) > self.min_attrs and self.rng.random() < 0.5:
                pos = int(self.rng.integers(0, len(ind["attrs"])))
                ind["attrs"].pop(pos)
                ind["values"].pop(pos)
            elif len(ind["attrs"]) < max_a:
                unused = list(set(range(n_features)) - set(ind["attrs"]))
                if unused:
                    new_attr   = int(self.rng.choice(unused))
                    insert_pos = int(np.searchsorted(ind["attrs"], new_attr))
                    ind["attrs"].insert(insert_pos, new_attr)
                    ind["values"].insert(insert_pos,
                                         int(self.rng.integers(0, 2)))

        if self.rng.random() < self.mutation_rate:
            ind["label"] = 1 - ind["label"]

        return ind

    # ── Rule conversion ───────────────────────────────────────────────────────

    def _to_rule(self, ind, Xn, y, bin_names, label_counts):
        attrs, values, label = ind["attrs"], ind["values"], ind["label"]
        mask    = np.all(Xn[:, attrs] == values, axis=1)
        covered = np.where(mask)[0]
        repet   = len(covered)
        if repet == 0:
            return None
        correct = int(np.sum(y[covered] == label))
        purity  = correct / repet
        if purity < self.purity or repet <= self.threshold:
            return None
        # Weight = fraction of all same-label training rows covered
        total_label = label_counts.get(label, 1)
        weight      = correct / total_label
        readable = [
            bin_names[a] if v == 1 else f"NOT ({bin_names[a]})"
            for a, v in zip(attrs, values)
        ]
        return {
            "label":    label,
            "attrs":    attrs,
            "values":   values,
            "purity":   float(purity),
            "repet":    repet,
            "weight":   float(weight),
            "readable": readable,
        }

    # ── Main fit ──────────────────────────────────────────────────────────────

    def fit(self, Xsel, y, original_feature_names):
        self.rules.clear()
        self.rng = np.random.default_rng(self.random_state)

        Xn         = Xsel.astype(int)
        n_features = Xn.shape[1]

        if self.verbose:
            print("\n=== Genetic Rule Miner Started ===")
            print(f"Selected binary matrix: {Xn.shape} | "
                  f"Generations: {self.n_generations} | "
                  f"Pop size: {self.pop_size} | "
                  f"Sharing sigma: {self.sharing_sigma}")

        # Readable names
        selected_cut_ids = self.selector.best_subset
        bin_names = []
        for cut_id in selected_cut_ids:
            if hasattr(self.binarizer, "cutpoint_readable"):
                bin_names.append(self.binarizer.cutpoint_readable(cut_id))
            else:
                feat_idx, thresh = self.binarizer.cutpoints[cut_id]
                bin_names.append(
                    f"{original_feature_names[feat_idx]} <= {thresh:.4f}"
                )

        # True class counts for weight denominator
        labels_u, counts_u = np.unique(y, return_counts=True)
        label_counts = dict(zip(labels_u.tolist(), counts_u.tolist()))

        population = np.array(
            [self._random_individual(n_features) for _ in range(self.pop_size)],
            dtype=object
        )
        raw_fitnesses = np.array(
            [self._fitness(ind, Xn, y, label_counts) for ind in population]
        )
        fitnesses = self._apply_sharing(population, raw_fitnesses)

        n_elite           = max(1, int(self.elite_frac * self.pop_size))
        best_fitness_ever = -1.0
        best_gen          = 0

        for gen in range(self.n_generations):
            # Elitism: preserve top individuals (ranked by raw fitness)
            elite_idx      = np.argsort(raw_fitnesses)[-n_elite:]
            elites         = population[elite_idx].tolist()
            new_population = elites[:]

            # Fill rest via tournament selection on shared fitness
            while len(new_population) < self.pop_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                c1, c2 = self._crossover(p1, p2, n_features)
                c1 = self._mutate(c1, n_features)
                c2 = self._mutate(c2, n_features)
                new_population.extend([c1, c2])

            population    = np.array(new_population[:self.pop_size], dtype=object)
            raw_fitnesses = np.array(
                [self._fitness(ind, Xn, y, label_counts) for ind in population]
            )
            fitnesses = self._apply_sharing(population, raw_fitnesses)

            gen_best = raw_fitnesses.max()
            if gen_best > best_fitness_ever:
                best_fitness_ever = gen_best
                best_gen          = gen

            if self.verbose and (gen + 1) % 50 == 0:
                print(f"  Gen {gen+1:4d}/{self.n_generations} | "
                      f"Best fitness: {gen_best:.4f} | "
                      f"Mean fitness: {raw_fitnesses.mean():.4f} | "
                      f"Non-zero: {(raw_fitnesses > 0).sum()}/{self.pop_size}")

        if self.verbose:
            print(f"\nEvolution complete. Best fitness {best_fitness_ever:.4f} "
                  f"first seen at generation {best_gen + 1}.")

        # Convert survivors to rules
        seen = {}
        for ind in population:
            if self._fitness(ind, Xn, y, label_counts) <= 0:
                continue
            rule = self._to_rule(ind, Xn, y, bin_names, label_counts)
            if rule is None:
                continue
            key = (rule["label"], tuple(rule["attrs"]), tuple(rule["values"]))
            if key not in seen or rule["weight"] > seen[key]["weight"]:
                seen[key] = rule

        self.rules = sorted(seen.values(),
                            key=lambda r: (-r["weight"], -r["purity"], r["label"]))

        if self.verbose:
            print(f"Generated {len(self.rules)} unique valid rules.\n")

    def print_rules(self, top_n=20):
        if not self.rules:
            print("No rules to display.")
            return
        print(f"\n=== TOP {top_n} RULES (sorted by weight) ===")
        for i, r in enumerate(self.rules[:top_n], 1):
            print(f"Rule {i:2d}: IF {' AND '.join(r['readable'])} "
                  f"THEN class = {r['label']} "
                  f"| Purity={r['purity']:.3f} "
                  f"| Support={r['repet']} "
                  f"| Weight={r['weight']:.3f}")