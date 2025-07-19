
Projeto: Otimização com Algoritmos Evolucionários
Este projeto foca no desenvolvimento, implementação e análise comparativa de uma Estratégia Evolutiva (EE) e um Algoritmo Genético (AG) para resolver problemas de otimização complexos. O objetivo é encontrar o ponto de mínimo global para um conjunto de quatro funções de benchmark bem conhecidas na literatura.

📚 Funções de Benchmark
O desempenho dos algoritmos foi avaliado na tarefa de minimizar as seguintes funções, utilizando uma dimensionalidade de d = 30 para todas:

Ackley Function

Rastrigin Function

Schwefel Function

Rosenbrock Function

Essas funções foram escolhidas por possuírem características desafiadoras, como múltiplos mínimos locais, vales estreitos e uma paisagem de busca complexa, testando a capacidade dos algoritmos de evitar ótimos locais e convergir para a solução global.

⚙️ Algoritmos Implementados
Foram implementados dois tipos de algoritmos evolucionários para comparação.

1. Estratégia Evolutiva (EE)
Foi implementada uma Estratégia Evolutiva do tipo (μ, λ) com auto-adaptação dos parâmetros de estratégia (sigmas).

1) Descrição esquemática do algoritmo:

Inicialização: Cria-se uma população inicial de μ indivíduos (pais).

Loop de Gerações:

Recombinação: λ filhos são gerados a partir da recombinação de pais selecionados aleatoriamente.

Mutação: Cada filho sofre mutação em seus parâmetros de estratégia (sigmas) e, em seguida, em suas variáveis de objeto (x).

Avaliação: O fitness de cada filho é calculado.

Seleção de Sobrevivência: Os μ melhores filhos se tornam a população de pais para a próxima geração, descartando a geração anterior (pais e filhos piores).

Término: O processo se repete até que a condição de parada (número máximo de gerações) seja atingida.

2) Detalhes da Implementação:

a. Representação das Soluções: Cada indivíduo é uma lista contendo três componentes:

vetor_x: Um vetor de d números de ponto flutuante, representando as variáveis da solução.

vetor_sigma: Um vetor de d valores σ, representando os parâmetros de estratégia (passos de mutação) para cada variável.

fitness: O valor da função objetivo para o vetor_x.

b. Função de Fitness: A própria função objetivo a ser minimizada (Ackley, Rastrigin, etc.). O objetivo é encontrar o indivíduo com o menor valor de fitness.

c. População:

Tamanho: Uma população de μ=30 pais que gera λ=200 filhos a cada geração.

Inicialização: As variáveis x são inicializadas aleatoriamente dentro dos limites de busca de cada função. Os sigmas são inicializados aleatoriamente em um intervalo (ex: [5.0, 15.0]) para garantir uma exploração inicial diversificada.

d. Processo de Seleção (para Recombinação): Para gerar cada filho, dois pais são selecionados de forma aleatória uniforme da população de μ pais.

e. Operadores Genéticos:

Recombinação:

Variáveis (x): Recombinação discreta, onde cada variável do filho é herdada de um dos dois pais com 50% de probabilidade.

Sigmas (σ): Recombinação intermediária, onde o sigma do filho é a média aritmética dos sigmas dos pais.

Mutação (Auto-Adaptativa):

Os sigmas são mutados primeiro, multiplicando-os por fatores log-normais (um global e um para cada dimensão). Isso permite que o algoritmo "aprenda" os melhores passos de mutação ao longo da evolução.

As variáveis x são então mutadas adicionando um ruído Gaussiano, cuja magnitude é controlada pelos sigmas recém-mutados (x' = x + σ' * N(0,1)).

f. Processo de Seleção por Sobrevivência: Uma seleção determinística (μ, λ). Apenas os μ melhores indivíduos da população de λ filhos são selecionados para formar a próxima geração de pais. Os pais da geração anterior são completamente descartados.

g. Condição de Término: O algoritmo para após um número fixo de gerações (MAX_GENERATIONS = 200).

2. Algoritmo Genético (AG)
(Esta seção descreve uma implementação geral, que deve ser adaptada para refletir os detalhes específicos do seu código de AG)

Um Algoritmo Genético canônico foi implementado para servir como base de comparação.

1) Descrição esquemática do algoritmo:

Inicialização: Cria-se uma população inicial de N indivíduos.

Loop de Gerações:

Avaliação: O fitness de cada indivíduo é calculado.

Seleção de Pais: Indivíduos são selecionados da população para reprodução com base em seu fitness.

Recombinação (Crossover): Pares de pais selecionados geram novos indivíduos (filhos).

Mutação: Os filhos gerados sofrem pequenas alterações aleatórias.

Seleção de Sobrevivência: A nova população é formada a partir dos pais e filhos (geralmente mantendo os melhores).

Término: O processo para quando a condição de parada é satisfeita.

2) Detalhes da Implementação:

a. Representação das Soluções: Um indivíduo é representado por um único vetor de d números de ponto flutuante (cromossomo), correspondendo às variáveis da solução x.

b. Função de Fitness: Similar à EE, a função objetivo é usada diretamente, mas frequentemente convertida para um problema de maximização (ex: 1 / (1 + fitness)) se forem usados métodos de seleção como a roleta.

c. População:

Tamanho: Uma população de tamanho fixo N.

Inicialização: Os indivíduos são inicializados com vetores x aleatórios dentro dos limites de busca de cada função.

d. Processo de Seleção (para Recombinação): Métodos como Seleção por Torneio ou Seleção por Roleta podem ser usados, dando aos indivíduos com melhor fitness uma maior probabilidade de serem selecionados para reprodução.

e. Operadores Genéticos:

Recombinação (Crossover): Para vetores de valores reais, operadores como Simulated Binary Crossover (SBX) ou Crossover Aritmético são adequados.

Mutação: Mutação Gaussiana, que adiciona um pequeno valor de uma distribuição normal a um ou mais genes (variáveis) do indivíduo.

f. Processo de Seleção por Sobrevivência: Uma abordagem comum é o Elitismo, onde os melhores k indivíduos da geração atual são copiados diretamente para a próxima geração, garantindo que o melhor resultado não seja perdido. O restante da nova população é preenchido pelos filhos gerados.

g. Condição de Término: Um número máximo de gerações ou um critério de estagnação (quando o melhor fitness não melhora por um certo número de gerações).

🚀 Como Executar o Código
Certifique-se de ter Python e a biblioteca numpy instalados.

Bash

pip install numpy
Abra o arquivo de script principal (ex: main.py).

No final do arquivo (dentro do bloco if __name__ == '__main__':), você pode configurar os seguintes parâmetros:

objective_func: A função que deseja otimizar (ex: ackley_function).

N_DIMS: O número de dimensões (definido como 30 para este projeto).

MU: Tamanho da população de pais (para EE).

LAMBDA: Número de filhos gerados (para EE).

MAX_GENERATIONS: Critério de parada.

Execute o script a partir do terminal:

Bash

python seu_arquivo.py
O progresso da otimização será impresso no console, mostrando o melhor fitness a cada geração.
