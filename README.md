# Otimizador de Corte CNC com Algoritmo Genético

## 1. Introdução

**Contextualização:**  
Este projeto aborda o problema de otimização de corte em máquinas CNC. O desafio consiste em dispor, de maneira eficiente, diferentes peças (com formatos variados, como retangulares, diamantes, circulares e triangulares) em uma chapa com dimensões fixas. Uma disposição otimizada reduz o desperdício de matéria‑prima, diminui o consumo de energia e minimiza o tempo de operação da máquina, contribuindo para a eficiência e competitividade operacional.

---

## 2. Escolha do Algoritmo Evolutivo

**Justificativa:**  
Optou‑se pelo Algoritmo Genético devido à sua robustez na exploração de espaços de solução combinatórios. Cada indivíduo é representado por uma permutação dos índices das peças, permitindo a exploração de diversas ordens de inserção. Essa abordagem, combinada com a estratégia de "Retângulos Livres" para a decodificação do layout, possibilita posicionar as peças sem redimensionamento, evitando sobreposições e cortes. A penalização severa de peças descartadas na função fitness incentiva o algoritmo a maximizar a alocação de peças dentro do bounding box fixo.

---

## 3. Descrição da Solução Implementada

### Arquitetura do Código

- **app.py:**  
  Define as dimensões da chapa e a lista de peças (_recortes_disponiveis_). Neste arquivo, são instanciados e executados os otimizadores evolutivos – neste caso, o Algoritmo Genético – e os layouts inicial e otimizado são exibidos.

- **genetic_algorithm.py:**  
  Contém a implementação do Algoritmo Genético com a estratégia de Retângulos Livres para posicionar as peças. A classe `GeneticAlgorithm` herda de `LayoutDisplayMixin` (do arquivo `common/layout_display.py`) para a visualização dos layouts. Os principais métodos incluem:
  - **initialize_population():** Gera uma população de indivíduos, onde cada indivíduo é uma permutação aleatória dos índices das peças.
  - **decode_layout():** Utiliza a abordagem de Retângulos Livres para posicionar as peças dentro da chapa, testando rotações (0° e 90°) e dividindo os espaços livres com o método `split_free_rect()`.
  - **get_dims():** Calcula as dimensões (largura e altura) de uma peça, considerando o tipo (retangular, diamante, circular, triangular) e a rotação.
  - **evaluate_individual():** Calcula o fitness de uma permutação com base na área do bounding box final do layout e penaliza fortemente as peças não alocadas.
  - **Operadores Genéticos:**  
    - **roulette_selection():** Seleciona indivíduos utilizando o método da roleta.  
    - **crossover_two_point():** Realiza o crossover de dois pontos para gerar um novo indivíduo.  
    - **mutate():** Realiza mutações por troca (swap) de posições na permutação, com uma taxa definida.
  - **run() e optimize_and_display():** Gerenciam o loop principal do GA e a exibição dos layouts inicial e otimizado.

### Modificações Realizadas

- **Inicialização da População:**  
  Evoluiu de um _stub_ para a geração efetiva de permutações aleatórias dos índices das peças.

- **Decodificação com Retângulos Livres:**  
  Implementou‑se uma estratégia para posicionar as peças sem redimensioná-las, utilizando a função `split_free_rect()` para atualizar os espaços livres e evitar sobreposições. A estratégia testa rotações de 0° e 90° para peças retangulares, diamantes e triangulares.

- **Suporte a Diferentes Formatos:**  
  O método `get_dims()` foi adaptado para calcular corretamente as dimensões para peças circulares, retangulares, diamantes e triangulares, considerando as rotações.

- **Função de Avaliação (Fitness):**  
  A função de fitness calcula a área do bounding box do layout final e adiciona uma penalidade forte para cada peça descartada, incentivando soluções que maximizem a alocação das peças.

- **Operadores Genéticos:**  
  Foram implementados:
  - Seleção via roleta, onde a chance de um indivíduo ser selecionado é proporcional a 1/(1 + fitness).
  - Crossover de dois pontos para recombinação de permutações.
  - Mutação por troca (swap) de posições, com uma taxa configurada (0.1).
  - Elitismo para preservar o melhor indivíduo entre as gerações.

### Ferramentas Utilizadas

- **Linguagem:** Python.
- **Bibliotecas Padrão:** `random`, `copy`, `math`, `typing`.
- **Mixin de Visualização:** `LayoutDisplayMixin` (arquivo `common/layout_display.py`).
- **ChatGPT:** Utilizado para suporte e discussão durante o desenvolvimento.

---

## 4. Resultados

### Disposição Inicial vs. Disposição Otimizada

- **Disposição Inicial:**  
  Apresenta as posições originais das peças, que podem incluir sobreposições e cortes.
  
- **Disposição Otimizada:**  
  Após a execução do GA, o layout final exibe as peças posicionadas sem sobreposição e sem cortes, respeitando as dimensões da chapa (bounding box fixo). Capturas de tela demonstram a melhoria na utilização do espaço.

### Impacto Econômico

- **Redução de Desperdício:**  
  A otimização permite um maior aproveitamento do material, diminuindo o desperdício de matéria‑prima.
- **Economia de Energia e Tempo:**  
  Um layout eficiente reduz o tempo de operação e os movimentos desnecessários da máquina, economizando energia.
- **Competitividade:**  
  A diminuição dos custos operacionais contribui para a competitividade no mercado.

### Tempo de Processamento

- O tempo de execução do algoritmo é adequado para aplicações práticas, sendo ajustável por meio dos parâmetros de população e gerações para balancear entre qualidade da solução e eficiência computacional.

---

## 5. Análise Crítica

### Vantagens e Desafios

- **Vantagens:**  
  - **Flexibilidade:** O algoritmo se adapta a diversos formatos de peças sem redimensioná-las.  
  - **Garantia de Não Sobreposição:** A abordagem de Retângulos Livres assegura que as peças sejam posicionadas sem sobreposição ou cortes.  
  - **Penalização Eficiente:** A função de fitness penaliza fortemente peças não alocadas, direcionando o GA para maximizar o aproveitamento do espaço.
  
- **Desafios:**  
  - **Precisão da Divisão dos Espaços Livres:** Pequenas imprecisões na função `split_free_rect()` podem ocasionar colisões ou subutilização do espaço.  
  - **Escalabilidade:** Em casos com um número elevado de peças, o tempo de processamento pode aumentar.  
  - **Limitações do Espaço Fixo:** A abordagem não utiliza multi‑chapas, o que pode penalizar soluções para conjuntos de peças que excedem a área disponível.

### Sugestões de Melhoria

- Refinar a função `split_free_rect()` para incluir uma margem de tolerância (epsilon) que evite erros de arredondamento.  
- Investigar operadores genéticos adaptativos e estratégias de crossover/mutação mais sofisticadas.  
- Considerar, em versões futuras, a implementação de uma abordagem multi‑chapas (bin packing) para conjuntos maiores de peças.  
- Integrar uma etapa de busca local para ajustes finos no layout otimizado.

---

## 6. Conclusão

### Resumo do Trabalho

O projeto implementa um Algoritmo Genético para otimização de cortes em CNC utilizando a estratégia de Retângulos Livres. O GA ignora as posições originais das peças, gerando um layout otimizado que garante a colocação sem sobreposições ou cortes, preservando as dimensões originais. A função de fitness avalia o aproveitamento do espaço e penaliza fortemente as peças descartadas, promovendo soluções eficientes que podem reduzir o desperdício e melhorar a eficiência operacional.

### Próximos Passos

- Refinar a precisão na divisão dos espaços livres com ajustes na função `split_free_rect()`.
- Explorar operadores genéticos adaptativos e a integração com métodos de busca local.
- Considerar a expansão do algoritmo para uma abordagem multi‑chapas em cenários com grande número de peças.
- Realizar testes em ambiente real para validar o impacto econômico e operacional da solução.

---

## Comandos para Instalação das Dependências

Certifique-se de ter o ambiente virtual configurado. Para instalar as dependências necessárias, execute:

```bash
pip freeze > requirements.txt
pip install -r requirements.txt
