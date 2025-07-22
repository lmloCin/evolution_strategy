import random
import math
import numpy as np
import matplotlib.pyplot as plt


def ackley_function(x):
    d = len(x)
    a = 20
    b = 0.2
    c = 2 * math.pi
    sum1 = sum(xi**2 for xi in x) 
    sum2 = sum(math.cos(c * xi) for xi in x)
    term1 = -a * math.exp(-b * math.sqrt(sum1 / d))
    term2 = -math.exp(sum2 / d)
    return term1 + term2 + a + math.e


def rastrigin_function(x):
    d = len(x)
    A = 10
    sum1 = sum(xi**2 - A * math.cos(2 * math.pi * xi) for xi in x)
    return A * d + sum1


def schwefel_function(x):
    d = len(x)
    sum1 = sum(xi * math.sin(math.sqrt(abs(xi))) for xi in x)
    return 418.9829 * d - sum1


def rosenbrock_function(x):
    d = len(x)
    sum1 = sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(d - 1))
    return sum1


def standard_deviation(indv):
    indv_minus_average_and_square = []
    for i in indv:
        indv_minus_average_and_square.append((i-(sum(indv)/len(indv)))**2)
    return (sum(indv_minus_average_and_square)/len(indv_minus_average_and_square))**0.5


def evolutionary_strategy(objective_func, n_dims, mu, lambda_, max_generations, func_name="Unknown Function"):
    """
    Implementa uma Estratégia Evolutiva (µ, λ) para minimizar uma função objetivo.

    Args:
        objective_func: A função a ser minimizada (ex: ackley_function).
        n_dims (int): O número de dimensões do problema.
        mu (int): O tamanho da população de pais.
        lambda_ (int): O número de filhos a serem gerados a cada geração.
        max_generations (int): O número de gerações para executar o algoritmo.
        func_name (str): Nome da função objetivo para o título do gráfico.

    Returns:
        tuple: O melhor indivíduo encontrado (vetor de variáveis) e seu valor de fitness.
    """

    # Define limites de busca específicos para cada função objetivo
    if objective_func == ackley_function:
        bounds = np.array([[-30.0, 30.0]] * n_dims)
    elif objective_func == rastrigin_function:
        bounds = np.array([[-5.12, 5.12]] * n_dims)
    elif objective_func == schwefel_function:
        bounds = np.array([[-500.0, 500.0]] * n_dims)
    elif objective_func == rosenbrock_function:
        bounds = np.array([[-5.0, 10.0]] * n_dims)
    else:
        raise ValueError("Função objetivo desconhecida. Defina os limites apropriados.")
    
    # Parâmetro para evitar que os passos de mutação fiquem muito pequenos 
    epsilon0 = 1e-3
    
    # Taxas de aprendizado para a mutação dos sigmas
    tau_prime = 1 / np.sqrt(2 * n_dims)
    tau = 1 / np.sqrt(2 * np.sqrt(n_dims))

    # Inicialização da população de 'mu' pais
    # Cada indivíduo é uma lista: [vetor_x, vetor_sigma, fitness]
    population = []
    for _ in range(mu):
        # Variáveis de objeto inicializadas aleatoriamente dentro dos limites
        x = np.random.uniform(bounds[:, 0], bounds[:, 1], n_dims)
        # Parâmetros de estratégia (sigmas) inicializados 
        sigma = np.random.uniform(5, 15, n_dims)
        fitness = objective_func(x)
        population.append([x, sigma, fitness])

    best_fitness_overall = float('inf')
    best_solution_overall = None
    
    generations_count = 0
    # Loop evolucionário principal
    generations_without_improvement = 0
    stagnation_limit = 100 # Critério de parada por estagnação
    
    # Histórico para gráficos
    best_fitness_history = []
    avg_fitness_history = []
    avg_sigma_history = [] # Para a média dos sigmas (parâmetros de mutação)

    # Loop evolucionário principal
    # O loop continua enquanto não atingir o máximo de gerações, e não houver estagnação.
    while generations_without_improvement < stagnation_limit and generations_count < max_generations: # Adicionado max_generations ao while
        generations_count += 1
        offspring = []
        for _ in range(lambda_):
            # 1. Seleção de Pais e Recombinação
            # Seleciona dois pais aleatoriamente da população 
            p1, p2 = np.random.choice(range(mu), 2, replace=False)
            parent1, parent2 = population[p1], population[p2]

            # Recombinação discreta para variáveis de objeto (x)
            x_child = np.array([parent1[0][i] if np.random.rand() < 0.5 else parent2[0][i] for i in range(n_dims)])

            # Recombinação intermediária para parâmetros de estratégia (sigma)
            sigma_child = (parent1[1] + parent2[1]) / 2.0

            # 2. Mutação (na ordem correta: sigma primeiro, depois x)
            # Mutação dos sigmas com distribuição log-normal (auto-adaptação)
            mutation_global = np.exp(tau_prime * np.random.normal(0, 1))
            mutations_individual = np.exp(tau * np.random.normal(0, 1, n_dims))
            sigma_child_mutated = sigma_child * mutation_global * mutations_individual

            # Garante que sigma não seja menor que um valor mínimo
            sigma_child_mutated = np.maximum(sigma_child_mutated, epsilon0)
            
            # Mutação das variáveis de objeto com ruído Gaussiano 
            x_child_mutated = x_child + sigma_child_mutated * np.random.normal(0, 1, n_dims)
            
            # Verificação de limites
            x_child_mutated = np.clip(x_child_mutated, bounds[:, 0], bounds[:, 1])

            # Avaliação do filho
            fitness_child = objective_func(x_child_mutated)
            offspring.append([x_child_mutated, sigma_child_mutated, fitness_child])

        # 3. Seleção de Sobreviventes: (µ, λ)
        # Ordena os filhos pelo fitness (minimização)
        offspring.sort(key=lambda item: item[2])
        
        # Os 'mu' melhores filhos se tornam a nova população de pais
        population = offspring[:mu]

        # Coleta de histórico para gráficos
        current_best_fitness = population[0][2]
        current_avg_fitness = np.mean([ind[2] for ind in population])
        current_avg_sigma = np.mean([np.mean(ind[1]) for ind in population]) # Média dos sigmas de todos os pais

        best_fitness_history.append(current_best_fitness)
        avg_fitness_history.append(current_avg_fitness)
        avg_sigma_history.append(current_avg_sigma) # Adiciona à lista

        # Atualiza melhor global e verifica estagnação
        if current_best_fitness < best_fitness_overall:
            best_fitness_overall = current_best_fitness
            best_solution_overall = population[0][0] # Armazena a solução x
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

    # --- Fim do Loop Evolucionário ---

    # 📊 Geração e SALVAMENTO dos Gráficos de Convergência
    plt.figure(figsize=(12, 5))

    # Gráfico 1: Melhor Fitness e Média de Fitness por Geração
    plt.subplot(1, 2, 1) # 1 linha, 2 colunas, 1º gráfico
    plt.plot(best_fitness_history, label="Melhor Fitness (Global)", color='blue')
    plt.plot(avg_fitness_history, label="Fitness Médio da População", color='red', linestyle='--')
    plt.title(f"Convergência do Fitness - {func_name}")
    plt.xlabel("Geração")
    plt.ylabel("Valor do Fitness")
    plt.grid(True)
    plt.legend()

    # Gráfico 2: Média dos Sigmas por Geração (parâmetros de mutação)
    plt.subplot(1, 2, 2) # 1 linha, 2 colunas, 2º gráfico
    plt.plot(avg_sigma_history, label="Média dos Sigmas", color='green')
    plt.title(f"Evolução da Média dos Sigmas - {func_name}")
    plt.xlabel("Geração")
    plt.ylabel("Valor Médio de Sigma")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout() # Ajusta o layout para evitar sobreposição
    plt.suptitle(f"Estratégia Evolutiva para {func_name} (μ={mu}, λ={lambda_}, D={n_dims})", y=1.02) # Título geral
    
    # SALVA O GRÁFICO EM VEZ DE MOSTRAR NA TELA
    plt.savefig(f"ES_convergencia_{func_name.lower().replace(' ', '_')}.png") 
    plt.close() # Fecha a figura para liberar memória (importante em loops de muitas execuções)


    print("\n--- Motivo da Parada ---")
    if generations_without_improvement >= stagnation_limit:
        print(f"Parada por estagnação: {stagnation_limit} gerações sem melhoria no fitness.")
    elif generations_count >= max_generations: # Verifica se atingiu max_generations
        print(f"Parada por atingir o número máximo de gerações ({max_generations}).")
    
    # Imprime o melhor resultado final da execução
    print(f"Resultado Final para {func_name}:")
    print(f"Gerações: {generations_count}")
    print(f"Melhor Fitness Encontrado: {best_fitness_overall:.8f}")
    print(f"Melhor Solução Encontrada (x): {best_solution_overall}")
    
    return best_solution_overall, best_fitness_overall, generations_count


if __name__ == '__main__':
    # Parâmetros da Estratégia Evolutiva
    N_DIMS = 30  # Número de dimensões
    MU = 50  # Tamanho da população (pais)
    LAMBDA = 1000 # Número de filhos gerados
    MAX_GENERATIONS = 5000 # Critério de parada de gerações

    print("Iniciando a Estratégia Evolutiva (µ, λ)-ES")

    # Lista de funções de benchmark a serem testadas
    benchmarks = [
        ("Ackley", ackley_function),
        ("Rastrigin", rastrigin_function),
        ("Schwefel", schwefel_function),
        ("Rosenbrock", rosenbrock_function),
    ]

    results_summary = [] # Para armazenar os resultados de cada execução para análise externa
    
    # Executa a ES para cada função de benchmark
    for name, func in benchmarks:
        print(f"\n" + "="*50)
        print(f"🏁 Otimizando função: {name}")
        
        # Realiza uma única execução da Estratégia Evolutiva
        best_solution_run, best_fitness_run, generations_count_run = evolutionary_strategy(
            objective_func=func, 
            n_dims=N_DIMS,
            mu=MU,
            lambda_=LAMBDA,
            max_generations=MAX_GENERATIONS,
            func_name=name # Passa o nome da função para o título do gráfico
        )
        # Salva o resultado para um resumo final, se necessário
        results_summary.append({
            "function": name,
            "best_solution": best_solution_run,
            "best_fitness": best_fitness_run,
            "generations": generations_count_run
        })
    
    print("\n" + "="*50)
    print("--- Resumo das Otimizações ---")
    for res in results_summary:
        print(f"Função: {res['function']}")
        print(f"  Melhor Fitness Final: {res['best_fitness']:.8f}")
        print(f"  Gerações Usadas: {res['generations']}")
        # print(f" Melhor Solução: {res['best_solution']}") # Descomentar para ver a solução X completa
    print("="*50)