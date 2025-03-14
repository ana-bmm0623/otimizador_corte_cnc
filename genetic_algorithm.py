"""
genetic_algorithm.py

Genetic Algorithm with Best-Fit Shelf Packing and Uniform Scaling

Features:
  - Ignores the user‑provided (x,y) positions and recalculates a layout via shelf packing.
  - Each individual is a permutation of the indices of the available pieces.
  - The shelf-packing decoder places pieces sequentially in shelves (no overlap, no pieces cut off).
  - If the overall layout (bounding box) exceeds the available sheet dimensions, the layout is uniformly scaled to fit.
  - For rectangular and diamond pieces, both 0° and 90° rotations are tested.
  - Fitness is computed as the area of the bounding box of the decoded layout (lower is better).
"""

from common.layout_display import LayoutDisplayMixin
import random
import copy
import math
from typing import List, Dict, Any, Tuple

class GeneticAlgorithm(LayoutDisplayMixin):
    def __init__(self, TAM_POP, recortes_disponiveis, sheet_width, sheet_height, numero_geracoes=100):
        """
        GA (Best-Fit Shelf Packing + Uniform Scaling) for Cutting Sheet Optimization.

        Parameters:
          - TAM_POP: Population size (e.g., 50, 100).
          - recortes_disponiveis: JSON-like list of pieces (types: retangular, diamante, circular, etc.).
          - sheet_width, sheet_height: Dimensions of the sheet.
          - numero_geracoes: Number of generations.
        
        This GA ignores user-provided positions and generates a layout that includes all pieces.
        After shelf packing, the entire layout is uniformly scaled to exactly fit the sheet,
        ensuring no overlap and no pieces are cut off.
        """
        print("GA (Best-Fit Shelf Packing + Scaling) - All pieces placed, no overlap or cuts.")
        self.TAM_POP = TAM_POP
        self.recortes_disponiveis = recortes_disponiveis
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.numero_geracoes = numero_geracoes

        # Population: each individual is a permutation of indices [0, 1, ..., n-1]
        self.POP: List[List[int]] = []
        self.best_individual: List[int] = []
        self.best_layout: List[Dict[str, Any]] = []
        self.best_fitness: float = float('inf')
        self.optimized_layout = None

        # GA parameters
        self.mutation_rate = 0.1
        self.elitism = True

        # Initialize population (random permutations)
        self.initialize_population()

    # -------------------------------------------------------------------------
    # 1. Initialize Population
    # -------------------------------------------------------------------------
    def initialize_population(self):
        """Creates TAM_POP random permutations of the indices of the available pieces."""
        n = len(self.recortes_disponiveis)
        base = list(range(n))
        for _ in range(self.TAM_POP):
            perm = base[:]
            random.shuffle(perm)
            self.POP.append(perm)

    # -------------------------------------------------------------------------
    # 2. Decoding: Best-Fit Shelf Packing with Uniform Scaling
    # -------------------------------------------------------------------------
    def decode_shelf_layout(self, permutation: List[int]) -> List[Dict[str, Any]]:
        """
        Decodes a permutation into a layout using a Best-Fit Shelf Packing strategy.
        The algorithm:
          - Starts with x_cursor = 0, y_cursor = 0, and shelf_height = 0.
          - For each piece (in the order given by the permutation):
              * For rectangular/diamond pieces, tests rotations 0° and 90° and chooses the first configuration that fits horizontally.
              * If x_cursor + piece_width exceeds sheet_width, starts a new shelf (x_cursor = 0; y_cursor += shelf_height; shelf_height = 0).
              * Places the piece at (x_cursor, y_cursor) and updates x_cursor and shelf_height.
          - After all pieces are placed, the bounding box of the layout is computed.
          - A uniform scaling factor is calculated to fit the entire layout within the sheet dimensions.
          - All piece positions and dimensions are scaled accordingly.
        This ensures that all pieces appear complete, with no overlapping or pieces cut off.
        """
        layout_result = []
        x_cursor = 0
        y_cursor = 0
        shelf_height = 0

        # Place all pieces in order, even if the layout exceeds the sheet
        for idx in permutation:
            rec_original = self.recortes_disponiveis[idx]

            # Determine possible configurations: for retangular/diamante, try 0° and 90°; otherwise, only 0°
            possible_configs = []
            if rec_original["tipo"] in ("retangular", "diamante"):
                for rot in [0, 90]:
                    r_copy = copy.deepcopy(rec_original)
                    r_copy["rotacao"] = rot
                    w, h = self.get_dims(r_copy)
                    possible_configs.append((rot, w, h))
            else:
                r_copy = copy.deepcopy(rec_original)
                r_copy["rotacao"] = 0
                w, h = self.get_dims(r_copy)
                possible_configs.append((0, w, h))

            # Choose the first configuration that fits horizontally
            best_config = None
            for (rot, w, h) in possible_configs:
                if w <= self.sheet_width:
                    best_config = (rot, w, h)
                    break
            if best_config is None:
                best_config = possible_configs[0]
            rot, w, h = best_config
            r_final = copy.deepcopy(rec_original)
            r_final["rotacao"] = rot

            # If the piece doesn't fit in the current shelf horizontally, start a new shelf
            if x_cursor + w > self.sheet_width:
                x_cursor = 0
                y_cursor += shelf_height
                shelf_height = 0

            # Place the piece at the current shelf position
            r_final["x"] = x_cursor
            r_final["y"] = y_cursor
            layout_result.append(r_final)
            x_cursor += w
            shelf_height = max(shelf_height, h)

        # Compute the bounding box of the layout
        x_min = min(rec["x"] for rec in layout_result)
        y_min = min(rec["y"] for rec in layout_result)
        x_max = max(rec["x"] + self.get_dims(rec)[0] for rec in layout_result)
        y_max = max(rec["y"] + self.get_dims(rec)[1] for rec in layout_result)
        layout_width = x_max - x_min
        layout_height = y_max - y_min

        # Calculate uniform scaling factor to fit the layout within the sheet
        scale_x = self.sheet_width / layout_width if layout_width > 0 else 1
        scale_y = self.sheet_height / layout_height if layout_height > 0 else 1
        scale = min(scale_x, scale_y)

        # Apply scaling to all pieces
        for rec in layout_result:
            rec["x"] = (rec["x"] - x_min) * scale
            rec["y"] = (rec["y"] - y_min) * scale
            if rec["tipo"] == "circular":
                rec["r"] = rec["r"] * scale
            elif rec["tipo"] in ("retangular", "diamante"):
                dims = self.get_dims(rec)
                # Update dimensions according to rotation
                if rec.get("rotacao", 0) == 90:
                    rec["altura"] = dims[0] * scale
                    rec["largura"] = dims[1] * scale
                else:
                    rec["largura"] = dims[0] * scale
                    rec["altura"] = dims[1] * scale
        return layout_result

    def get_dims(self, rec: Dict[str, Any]) -> Tuple[float, float]:
        """Returns (w, h) for the piece, considering rotation (0° or 90°)."""
        tipo = rec["tipo"]
        if tipo == "circular":
            d = 2 * rec["r"]
            return d, d
        elif tipo in ("retangular", "diamante"):
            if rec.get("rotacao", 0) == 90:
                return rec["altura"], rec["largura"]
            else:
                return rec["largura"], rec["altura"]
        else:
            return rec.get("largura", 10), rec.get("altura", 10)

    def get_layout_box_placed(self, rec: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """Returns (x, y, w, h) of the placed piece for display."""
        x, y = rec["x"], rec["y"]
        w, h = self.get_dims(rec)
        return (x, y, w, h)

    # -------------------------------------------------------------------------
    # 3. Evaluation (Fitness)
    # -------------------------------------------------------------------------
    def evaluate_individual(self, permutation: List[int]) -> float:
        """
        Decodes the permutation into a layout via shelf packing and computes the fitness
        as the area of the bounding box of the final layout. Lower area indicates better space utilization.
        """
        layout = self.decode_shelf_layout(permutation)
        if not layout:
            return self.sheet_width * self.sheet_height * 2  # Penalize if layout is empty

        x_min = float('inf')
        x_max = float('-inf')
        y_min = float('inf')
        y_max = float('-inf')
        for rec in layout:
            bx, by, bw, bh = self.get_layout_box_placed(rec)
            x_min = min(x_min, bx)
            x_max = max(x_max, bx + bw)
            y_min = min(y_min, by)
            y_max = max(y_max, by + bh)
        area_layout = (x_max - x_min) * (y_max - y_min)
        return area_layout

    def evaluate_population(self):
        for perm in self.POP:
            fit = self.evaluate_individual(perm)
            if fit < self.best_fitness:
                self.best_fitness = fit
                self.best_individual = perm[:]

    # -------------------------------------------------------------------------
    # 4. Genetic Operators (Permutation-based)
    # -------------------------------------------------------------------------
    def compute_fitness_scores(self) -> List[float]:
        fits = [self.evaluate_individual(perm) for perm in self.POP]
        return [1 / (1 + f) for f in fits]

    def roulette_selection(self) -> List[int]:
        scores = self.compute_fitness_scores()
        total = sum(scores)
        pick = random.random() * total
        current = 0
        for perm, sc in zip(self.POP, scores):
            current += sc
            if current >= pick:
                return perm
        return self.POP[-1]

    def crossover_two_point(self, p1: List[int], p2: List[int]) -> List[int]:
        """2-point order crossover for permutations."""
        size = len(p1)
        i1, i2 = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[i1:i2 + 1] = p1[i1:i2 + 1]
        p2_idx = 0
        for i in range(size):
            if child[i] is None:
                while p2[p2_idx] in child:
                    p2_idx += 1
                child[i] = p2[p2_idx]
                p2_idx += 1
        return child

    def mutate(self, perm: List[int]) -> List[int]:
        """Mutation: swap two positions in the permutation."""
        if random.random() < self.mutation_rate:
            i1, i2 = random.sample(range(len(perm)), 2)
            perm[i1], perm[i2] = perm[i2], perm[i1]
        return perm

    def genetic_operators(self):
        new_pop = []
        if self.elitism and self.best_individual:
            new_pop.append(self.best_individual[:])
        while len(new_pop) < self.TAM_POP:
            p1 = self.roulette_selection()
            p2 = self.roulette_selection()
            child = self.crossover_two_point(p1, p2)
            child = self.mutate(child)
            new_pop.append(child)
        self.POP = new_pop[:self.TAM_POP]

    # -------------------------------------------------------------------------
    # 5. Main Loop
    # -------------------------------------------------------------------------
    def run(self):
        for gen in range(self.numero_geracoes):
            self.evaluate_population()
            self.genetic_operators()
            if gen % 10 == 0:
                print(f"Generation {gen} - Best Fitness: {self.best_fitness}")
        self.best_layout = self.decode_shelf_layout(self.best_individual)
        self.optimized_layout = self.best_layout
        return self.best_layout

    def optimize_and_display(self):
        """
        Displays the initial layout (user-provided positions) and the final optimized layout
        (calculated via Best-Fit Shelf Packing and scaled to fit).
        """
        self.display_layout(self.recortes_disponiveis, title="Initial Layout - GA (Best-Fit Shelf)")
        self.run()
        self.display_layout(self.optimized_layout, title="Optimized Layout - GA (Best-Fit Shelf)")
        return self.optimized_layout