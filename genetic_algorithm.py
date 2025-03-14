"""
genetic_algorithm.py

Algoritmo Genético com Shelf Packing Multi‑Chapa (Bin Packing)
- Ignora as posições (x, y) originais; o layout é recalculado.
- Cada indivíduo é uma permutação dos índices das peças.
- A decodificação utiliza Shelf Packing, retornando também as peças que não couberam na chapa atual.
- Se houver peças não alocadas, uma segunda (ou mais) chapa é gerada (o layout final é a combinação vertical dos layouts de cada chapa).
- Para peças do tipo "retangular" e "diamante", testa rotações 0° e 90° e escolhe a configuração que melhor se encaixa.
- O fitness é calculado penalizando fortemente cada peça não colocada.
"""

from common.layout_display import LayoutDisplayMixin
import random
import copy
import math
from typing import List, Dict, Any, Tuple

class GeneticAlgorithm(LayoutDisplayMixin):
    def __init__(self, TAM_POP, recortes_disponiveis, sheet_width, sheet_height, numero_geracoes=100):
        """
        GA Multi‑Chapa para Otimização de Corte (bin packing sem redimensionamento).

        Parâmetros:
          - TAM_POP: Tamanho da população (ex.: 50, 100).
          - recortes_disponiveis: Lista de peças (JSON-like, com tipos: retangular, diamante, circular, etc.).
          - sheet_width, sheet_height: Dimensões da chapa.
          - numero_geracoes: Número de gerações.
        """
        print("GA Multi‑Chapa (Bin Packing) - Otimizando disposição sem sobreposições e sem cortes.")
        self.TAM_POP = TAM_POP
        self.recortes_disponiveis = recortes_disponiveis
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.numero_geracoes = numero_geracoes

        # Cada indivíduo é uma permutação dos índices [0 ... n-1]
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
        """Cria TAM_POP permutações aleatórias dos índices das peças."""
        n = len(self.recortes_disponiveis)
        base = list(range(n))
        for _ in range(self.TAM_POP):
            perm = base[:]
            random.shuffle(perm)
            self.POP.append(perm)

    # -------------------------------------------------------------------------
    # 2. Decodificação com Shelf Packing Multi‑Chapa
    # -------------------------------------------------------------------------
    def decode_shelf_layout(self, permutation: List[int]) -> Tuple[List[Dict[str, Any]], List[int]]:
        """
        Decodifica uma permutação em um layout para UMA chapa usando Shelf Packing.
        Retorna uma tupla (layout_result, leftover), onde:
          - layout_result: lista de peças posicionadas na chapa atual (sem sobreposição ou cortes).
          - leftover: lista de índices de peças que não couberam verticalmente.
        """
        layout_result = []
        leftover = []
        x_cursor = 0
        y_cursor = 0
        shelf_height = 0

        for idx in permutation:
            rec_original = self.recortes_disponiveis[idx]
            # Testa configurações possíveis: para retangular/diamante, testa 0° e 90°; para outros, só 0°
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
            # Seleciona a primeira configuração que cabe horizontalmente
            best_config = None
            for (rot, w, h) in possible_configs:
                if w <= self.sheet_width:
                    best_config = (rot, w, h)
                    break
            if best_config is None:
                leftover.append(idx)
                continue

            rot, w, h = best_config
            r_final = copy.deepcopy(rec_original)
            r_final["rotacao"] = rot

            # Se não couber horizontalmente na shelf atual, inicia nova shelf
            if x_cursor + w > self.sheet_width:
                x_cursor = 0
                y_cursor += shelf_height
                shelf_height = 0

            # Se não couber verticalmente, adiciona aos leftovers
            if y_cursor + h > self.sheet_height:
                leftover.append(idx)
                continue

            # Posiciona a peça
            r_final["x"] = x_cursor
            r_final["y"] = y_cursor
            layout_result.append(r_final)
            x_cursor += w
            shelf_height = max(shelf_height, h)

        return (layout_result, leftover)

    def decode_multisheet(self, permutation: List[int]) -> List[Dict[str, Any]]:
        """
        Decodifica uma permutação em um layout multi‑chapa.
        Enquanto houver peças (índices) não colocadas, chama decode_shelf_layout e
        empilha verticalmente cada chapa (com um gap de 10 unidades entre elas).
        Retorna um layout único, com as peças de todas as chapas ajustadas verticalmente.
        """
        total_layout = []
        current_perm = permutation[:]
        offset = 0
        gap = 10  # Espaço entre chapas
        while current_perm:
            layout, leftover = self.decode_shelf_layout(current_perm)
            # Ajusta a coordenada y de cada peça nesta chapa
            for rec in layout:
                rec["y"] += offset
            total_layout.extend(layout)
            # Se nenhuma peça foi colocada nesta chapa, interrompe
            if not layout:
                break
            # Calcula a altura utilizada nesta chapa
            sheet_height_used = 0
            for rec in layout:
                _, y, w, h = self.get_layout_box_placed(rec)
                sheet_height_used = max(sheet_height_used, y + h)
            offset += sheet_height_used + gap
            current_perm = leftover
        return total_layout

    def get_dims(self, rec: Dict[str, Any]) -> Tuple[float, float]:
        """Retorna (w, h) considerando a rotação (0° ou 90°)."""
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
        """Retorna (x, y, w, h) para a peça posicionada, considerando rotação."""
        x, y = rec["x"], rec["y"]
        w, h = self.get_dims(rec)
        return (x, y, w, h)

    # -------------------------------------------------------------------------
    # 3. Avaliação (Fitness)
    # -------------------------------------------------------------------------
    def evaluate_individual(self, permutation: List[int]) -> float:
        """
        Decodifica a permutação usando multi‑chapa e retorna o fitness,
        que é a área do bounding box que envolve todas as chapas.
        Além disso, penaliza fortemente cada peça que não for alocada.
        """
        layout, leftover = self.decode_shelf_layout(permutation)
        multi_sheet_layout = self.decode_multisheet(permutation)
        # Se alguma peça não foi colocada na primeira chapa, adiciona penalidade
        penalty = 100000 * len(leftover)
        if not multi_sheet_layout:
            return self.sheet_width * self.sheet_height * 2
        x_min = float('inf')
        x_max = float('-inf')
        y_min = float('inf')
        y_max = float('-inf')
        for rec in multi_sheet_layout:
            bx, by, bw, bh = self.get_layout_box_placed(rec)
            x_min = min(x_min, bx)
            x_max = max(x_max, bx + bw)
            y_min = min(y_min, by)
            y_max = max(y_max, by + bh)
        area_layout = (x_max - x_min) * (y_max - y_min)
        return area_layout + penalty

    def evaluate_population(self):
        """Avalia cada indivíduo e atualiza o melhor encontrado."""
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
        """Crossover de dois pontos para permutações."""
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
        """Mutação: troca de duas posições na permutação."""
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
        # Decodifica a melhor permutação usando multi-sheet decode
        self.best_layout = self.decode_multisheet(self.best_individual)
        self.optimized_layout = self.best_layout
        return self.best_layout

    def optimize_and_display(self):
        """
        Exibe o layout inicial (posições originais fornecidas) e o layout final otimizado
        (decodificado via multi‑chapa).
        """
        self.display_layout(self.recortes_disponiveis, title="Initial Layout - GA (Best-Fit Shelf)")
        self.run()
        self.display_layout(self.optimized_layout, title="Optimized Layout - GA (Best-Fit Shelf)")
        return self.optimized_layout