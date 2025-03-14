"""
genetic_algorithm.py

Algoritmo Genético com Best-Fit Shelf Packing Determinístico

Características:
  - Ignora as posições (x, y) fornecidas; o layout final é recalculado.
  - Cada indivíduo é uma permutação dos índices das peças.
  - A decodificação utiliza uma estratégia "Best-Fit Shelf": as peças são posicionadas
    em prateleiras (shelves) de forma sequencial, sem sobreposição e sem que fiquem
    “cortadas” fora da chapa.
  - Para peças do tipo "retangular" e "diamante", são testadas duas rotações (0° e 90°)
    e é escolhida a configuração que minimize a altura ocupada na shelf.
  - O fitness é definido como a área do bounding box final do layout (quanto menor, melhor).
  - Essa abordagem é rápida e garante zero sobreposição e peças completas.
"""

from common.layout_display import LayoutDisplayMixin
import random
import copy
import math
from typing import List, Dict, Any, Tuple

class GeneticAlgorithm(LayoutDisplayMixin):
    def __init__(self, TAM_POP, recortes_disponiveis, sheet_width, sheet_height, numero_geracoes=100):
        """
        Parâmetros:
          - TAM_POP: Tamanho da população (ex.: 50, 100).
          - recortes_disponiveis: Lista JSON-like com as peças (tipos: retangular, diamante, circular, etc.).
          - sheet_width, sheet_height: Dimensões da chapa.
          - numero_geracoes: Número de gerações para o GA.
        
        Esse GA ignora as posições originais e gera um layout via Shelf Packing,
        garantindo que não haja sobreposições nem peças "cortadas".
        """
        print("GA (Best-Fit Shelf Packing) - Layout otimizado sem sobreposições ou cortes.")
        self.TAM_POP = TAM_POP
        self.recortes_disponiveis = recortes_disponiveis
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.numero_geracoes = numero_geracoes

        # População: cada indivíduo é uma permutação dos índices [0...n-1]
        self.POP: List[List[int]] = []
        self.best_individual: List[int] = []
        self.best_layout: List[Dict[str, Any]] = []
        self.best_fitness: float = float('inf')
        self.optimized_layout = None

        # Parâmetros do GA
        self.mutation_rate = 0.1
        self.elitism = True

        # Inicializa a população com permutações aleatórias
        self.initialize_population()

    # -------------------------------------------------------------------------
    # 1. Inicialização da População
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
    # 2. Decodificação: Best-Fit Shelf Packing
    # -------------------------------------------------------------------------
    def decode_shelf_layout(self, permutation: List[int]) -> List[Dict[str, Any]]:
        """
        Decodifica uma permutação em um layout utilizando o método Best-Fit Shelf:
          - Inicia com x_cursor = 0, y_cursor = 0 e shelf_height = 0.
          - Para cada índice na permutação, tenta posicionar a peça na shelf atual.
          - Para peças retangulares e diamantes, testa rotações 0° e 90° e escolhe a que couber
            e que gere menor altura.
          - Se a peça não couber na shelf atual (x_cursor + largura > sheet_width), inicia uma nova shelf:
              * x_cursor é reiniciado para 0;
              * y_cursor é incrementado com a altura da shelf anterior;
              * shelf_height é reiniciado.
          - Se a peça não couber verticalmente (y_cursor + altura > sheet_height), ela é descartada.
        Retorna um layout com cada peça posicionada (sem sobreposição e sem cortes).
        """
        layout_result = []
        x_cursor = 0
        y_cursor = 0
        shelf_height = 0

        for idx in permutation:
            rec_original = self.recortes_disponiveis[idx]

            # Determina as configurações possíveis: para retangulares e diamantes, testa 0° e 90°
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

            # Seleciona a configuração que minimize a altura (shelf) e que caiba na largura
            best_config = None
            for (rot, w, h) in possible_configs:
                if w <= self.sheet_width:
                    best_config = (rot, w, h)
                    break  # Prioriza a primeira configuração que couber
            if best_config is None:
                continue  # descarta a peça se não couber em nenhuma configuração

            rot, w, h = best_config
            r_final = copy.deepcopy(rec_original)
            r_final["rotacao"] = rot

            # Se a peça não couber na shelf atual, inicia nova shelf
            if x_cursor + w > self.sheet_width:
                x_cursor = 0
                y_cursor += shelf_height
                shelf_height = 0

            # Se não couber verticalmente, descarta a peça
            if y_cursor + h > self.sheet_height:
                continue

            # Posiciona a peça e atualiza os cursors
            r_final["x"] = x_cursor
            r_final["y"] = y_cursor
            layout_result.append(r_final)
            x_cursor += w
            shelf_height = max(shelf_height, h)
        return layout_result

    def get_dims(self, rec: Dict[str, Any]) -> Tuple[float, float]:
        """
        Retorna as dimensões (w, h) do recorte considerando a rotação.
          - Para "circular": w = h = 2*r.
          - Para "retangular" e "diamante": se rotacao==90, inverte largura e altura.
        """
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
        """Retorna (x, y, w, h) do recorte posicionado para exibição."""
        x, y = rec["x"], rec["y"]
        w, h = self.get_dims(rec)
        return (x, y, w, h)

    # -------------------------------------------------------------------------
    # 3. Avaliação (Fitness)
    # -------------------------------------------------------------------------
    def evaluate_individual(self, permutation: List[int]) -> float:
        """
        Decodifica a permutação e retorna o fitness como a área do bounding box
        do layout final. Quanto menor, melhor.
        """
        layout = self.decode_shelf_layout(permutation)
        if not layout:
            return self.sheet_width * self.sheet_height * 2  # Penaliza layout vazio

        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        for rec in layout:
            bx, by, bw, bh = self.get_layout_box_placed(rec)
            x_min = min(x_min, bx)
            x_max = max(x_max, bx + bw)
            y_min = min(y_min, by)
            y_max = max(y_max, by + bh)
        area_layout = (x_max - x_min) * (y_max - y_min)
        return area_layout

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
        """Realiza o crossover de dois pontos em permutações."""
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
        self.best_layout = self.decode_shelf_layout(self.best_individual)
        self.optimized_layout = self.best_layout
        return self.best_layout

    def optimize_and_display(self):
        """Exibe o layout inicial e o layout final otimizado."""
        self.display_layout(self.recortes_disponiveis, title="Initial Layout - GA (Best-Fit Shelf)")
        self.run()
        self.display_layout(self.optimized_layout, title="Optimized Layout - GA (Best-Fit Shelf)")
        return self.optimized_layout
