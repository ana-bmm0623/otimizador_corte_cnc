"""
genetic_algorithm.py

Algoritmo Genético com Colocação 2D via Retângulos Livres (Free Rectangles):
 - Não altera o bounding box (permanece sheet_width x sheet_height).
 - Não redimensiona peças.
 - Não usa multi-chapas.
 - Se a peça não couber em nenhum retângulo livre (mesmo girando 0° ou 90°),
   ela é descartada e penalizada.
 - Suporta rotações 0°/90° para peças retangulares, diamantes e triangulares.
 - Penaliza fortemente as peças não colocadas.
 - Adaptável a diferentes formatos: retangulares, diamantes, circulares, triangulares etc.,
   sem sobreposições e sem cortar imagens.
"""

from common.layout_display import LayoutDisplayMixin
import random
import copy
import math
from typing import List, Dict, Any, Tuple

class GeneticAlgorithm(LayoutDisplayMixin):
    def __init__(
        self,
        TAM_POP: int,
        recortes_disponiveis: List[Dict[str, Any]],
        sheet_width: float,
        sheet_height: float,
        numero_geracoes: int = 100
    ):
        """
        GA com Retângulos Livres para colocação 2D:
         - Sem multi-chapas, sem redimensionar peças.
         - Se a peça não couber em nenhum retângulo livre, ela é descartada (penalidade).
         - Testa rotações 0°/90° para retangulares, diamantes e triangulares.
        """
        print("GA Free-Rectangles - Otimização do Corte CNC sem sobreposições e cortes.")
        self.TAM_POP = TAM_POP
        self.recortes_disponiveis = recortes_disponiveis
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.numero_geracoes = numero_geracoes

        # Cada indivíduo é uma permutação dos índices [0..n-1]
        self.POP: List[List[int]] = []
        self.best_individual: List[int] = []
        self.best_fitness: float = float('inf')
        self.optimized_layout = None

        # Parâmetros do GA
        self.mutation_rate = 0.1
        self.elitism = True

        self.initialize_population()

    # -------------------------------------------------------------------------
    # 1. Inicialização da População (Permutações)
    # -------------------------------------------------------------------------
    def initialize_population(self):
        n = len(self.recortes_disponiveis)
        base = list(range(n))
        for _ in range(self.TAM_POP):
            perm = base[:]
            random.shuffle(perm)
            self.POP.append(perm)

    # -------------------------------------------------------------------------
    # 2. Decodificação via Retângulos Livres
    # -------------------------------------------------------------------------
    def decode_layout(self, permutation: List[int]) -> Tuple[List[Dict[str, Any]], int]:
        """
        Decodifica a permutação em layout usando a abordagem de Retângulos Livres (Free Rectangles).
        - Inicia com um retângulo livre do tamanho da chapa: (0, 0, sheet_width, sheet_height).
        - Para cada peça, testa configurações possíveis (para retangular, diamante e triangular, testa 0° e 90°).
        - Se a peça couber totalmente em um retângulo livre, ela é posicionada e o retângulo livre é dividido
          (utilizando a função split_free_rect).
        - Se a peça não couber em nenhum retângulo livre, ela é descartada e penalizada.
        Retorna (layout_result, num_descartadas).
        """
        layout_result: List[Dict[str, Any]] = []
        free_rects: List[Tuple[float, float, float, float]] = []
        free_rects.append((0, 0, self.sheet_width, self.sheet_height))
        discarded = 0

        for idx in permutation:
            rec = self.recortes_disponiveis[idx]
            possible_configs = []
            if rec["tipo"] in ("retangular", "diamante", "triangular"):
                for rot in [0, 90]:
                    w, h = self.get_dims(rec, rot)
                    possible_configs.append((rot, w, h))
            else:
                w, h = self.get_dims(rec, 0)
                possible_configs.append((0, w, h))

            placed = False
            for (rot, w, h) in possible_configs:
                best_index = -1
                for i, (rx, ry, rw, rh) in enumerate(free_rects):
                    if w <= rw and h <= rh:
                        best_index = i
                        break
                if best_index != -1:
                    placed = True
                    r_final = copy.deepcopy(rec)
                    r_final["rotacao"] = rot
                    (rx, ry, rw, rh) = free_rects[best_index]
                    r_final["x"] = rx
                    r_final["y"] = ry
                    layout_result.append(r_final)
                    del free_rects[best_index]
                    new_rects = self.split_free_rect((rx, ry, rw, rh), (rx, ry, w, h))
                    free_rects.extend(new_rects)
                    break
            if not placed:
                discarded += 1

        return (layout_result, discarded)

    def split_free_rect(self, free_rect: Tuple[float, float, float, float],
                        placed_rect: Tuple[float, float, float, float]) -> List[Tuple[float, float, float, float]]:
        """
        Divide o retângulo livre (free_rect) removendo a área ocupada pelo placed_rect.
        Retorna uma lista de novos retângulos livres que não se sobrepõem.
        """
        fx, fy, fw, fh = free_rect
        px, py, pw, ph = placed_rect
        new_rects = []
        # Se não houver interseção, retorna o free_rect original
        if px >= fx + fw or px + pw <= fx or py >= fy + fh or py + ph <= fy:
            return [free_rect]

        # Calcula a interseção
        ix = max(fx, px)
        iy = max(fy, py)
        ix2 = min(fx + fw, px + pw)
        iy2 = min(fy + fh, py + ph)

        # Retângulo à esquerda
        if ix > fx:
            new_rects.append((fx, fy, ix - fx, fh))
        # Retângulo à direita
        if ix2 < fx + fw:
            new_rects.append((ix2, fy, (fx + fw) - ix2, fh))
        # Retângulo acima (na área de interseção horizontal)
        if iy > fy:
            new_rects.append((ix, fy, ix2 - ix, iy - fy))
        # Retângulo abaixo
        if iy2 < fy + fh:
            new_rects.append((ix, iy2, ix2 - ix, (fy + fh) - iy2))

        return new_rects

    def get_dims(self, rec: Dict[str, Any], rot: int) -> Tuple[float, float]:
        """
        Retorna (w, h) considerando o tipo da peça e a rotação.
          - Para "circular": w = h = 2 * r.
          - Para "retangular" e "diamante": se rot == 90, inverte largura e altura.
          - Para "triangular": assume que o recorte possui "b" (base) e "h" (altura); se rot == 90, retorna (h, b).
        """
        tipo = rec["tipo"]
        if tipo == "circular":
            d = 2 * rec["r"]
            return (d, d)
        elif tipo == "triangular":
            if rot == 90:
                return (rec["h"], rec["b"])
            else:
                return (rec["b"], rec["h"])
        elif tipo in ("retangular", "diamante"):
            if rot == 90:
                return (rec["altura"], rec["largura"])
            else:
                return (rec["largura"], rec["altura"])
        else:
            return (rec.get("largura", 10), rec.get("altura", 10))

    def get_layout_box_placed(self, rec: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """Retorna (x, y, w, h) da peça posicionada, considerando a rotação."""
        x, y = rec["x"], rec["y"]
        rot = rec.get("rotacao", 0)
        w, h = self.get_dims(rec, rot)
        return (x, y, w, h)

    # -------------------------------------------------------------------------
    # 3. Avaliação (Fitness)
    # -------------------------------------------------------------------------
    def evaluate_individual(self, permutation: List[int]) -> float:
        """
        Decodifica a permutação usando free rectangles e retorna o fitness:
         - Área do bounding box final do layout
         - Penalidade para cada peça descartada.
        """
        layout, discarded = self.decode_layout(permutation)
        if not layout:
            return self.sheet_width * self.sheet_height * 2 + discarded * 10000

        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        for rec in layout:
            bx, by, bw, bh = self.get_layout_box_placed(rec)
            x_min = min(x_min, bx)
            x_max = max(x_max, bx + bw)
            y_min = min(y_min, by)
            y_max = max(y_max, by + bh)
        area_layout = (x_max - x_min) * (y_max - y_min)
        penalty = discarded * 10000
        return area_layout + penalty

    def evaluate_population(self):
        for perm in self.POP:
            fit = self.evaluate_individual(perm)
            if fit < self.best_fitness:
                self.best_fitness = fit
                self.best_individual = perm[:]

    # -------------------------------------------------------------------------
    # 4. Operadores Genéticos (Permutações)
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
    # 5. Loop Principal
    # -------------------------------------------------------------------------
    def run(self):
        for gen in range(self.numero_geracoes):
            self.evaluate_population()
            self.genetic_operators()
            if gen % 10 == 0:
                print(f"Geração {gen} - Melhor Fitness: {self.best_fitness}")
        layout, _ = self.decode_layout(self.best_individual)
        self.optimized_layout = layout
        return self.optimized_layout

    def optimize_and_display(self):
        """
        Exibe o layout inicial (com posições originais) e o layout final otimizado
        usando free rectangles (sem redimensionar ou multi-chapas).
        """
        self.display_layout(self.recortes_disponiveis, title="Initial Layout - GA (FreeRect)")
        self.run()
        self.display_layout(self.optimized_layout, title="Optimized Layout - GA (FreeRect)")
        return self.optimized_layout
