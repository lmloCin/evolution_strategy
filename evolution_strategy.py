import random
import math
import numpy as np

def ackley_function(x):
    d = len(x)
    a = 20
    b = 0.2
    c = 2 * 3.14159265358979323846
    sum1 = sum(xi**2 for xi in x) 
    sum2 = sum(math.cos(c * xi) for xi in x)
    term1 = -a * math.exp(-b * math.sqrt(sum1 / d))
    term2 =  -math.exp(sum2 / d)
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

def evolutionary_strategy(objective_func, n_dims, mu, lambda_, max_generations):
    """
    Implementa uma Estratégia Evolutiva (µ, λ) para minimizar uma função objetivo.

    Args:
        objective_func: A função a ser minimizada (ex: ackley_function).
        n_dims (int): O número de dimensões do problema.
        mu (int): O tamanho da população de pais.
        lambda_ (int): O número de filhos a serem gerados a cada geração.
        max_generations (int): O número de gerações para executar o algoritmo.

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

    # Loop evolucionário principal
    for gen in range(max_generations):
        offspring = []
        for _ in range(lambda_):
            # 1. Seleção de Pais e Recombinação
            # Seleciona dois pais aleatoriamente da população 
            p1, p2 = np.random.choice(range(mu), 2, replace=False)
            parent1, parent2 = population[p1], population[p2]

            # Recombinação discreta para variáveis de objeto (x) 
            x_child = np.array([parent1[0][i] if np.random.rand() < 0.5 else parent2[0][i] for i in range(n_dims)])
            
            # Recombinação intermediária para parâmetros de estratégia (sigma)

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

        current_best_fitness = population[0][2]
        if current_best_fitness < best_fitness_overall:
            best_fitness_overall = current_best_fitness
            best_solution_overall = population[0][0]

        if (gen + 1) % 100 == 0 or gen < 10:
            print(f"Geração {gen+1:4d}: Melhor Fitness = {best_fitness_overall:.8f} | Melhor Solução = {population[0][0]}")

    return best_solution_overall, best_fitness_overall

if __name__ == '__main__':
    # Parâmetros da Estratégia Evolutiva
    N_DIMS = 3           # Número de dimensões, como no exemplo da função de Ackley
    MU = 30               # Tamanho da população (pais)
    LAMBDA = 200          # Número de filhos gerados
    MAX_GENERATIONS = 1000 # Critério de parada

    print("Iniciando a Estratégia Evolutiva (30, 200) para a Função de Ackley...")
    
    best_solution, best_fitness = evolutionary_strategy(
        objective_func=rosenbrock_function,
        n_dims=N_DIMS,
        mu=MU,
        lambda_=LAMBDA,
        max_generations=MAX_GENERATIONS
    )

    print("\nOtimização concluída.")
    print(f"Melhor solução encontrada: {best_solution}")
    print(f"Valor mínimo da função (fitness): {best_fitness}")