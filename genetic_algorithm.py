import numpy as np
import math
import random
import matplotlib.pyplot as plt  # 📌 biblioteca para gráfico


# ... (Funções de benchmark: ackley_function, etc. — mantenha como no seu código original)
# --- Funções de benchmark (mantidas) ---

def ackley_function(x):
    d = len(x)
    a = 20
    b = 0.2
    c = 2 * math.pi
    sum1 = sum(xi ** 2 for xi in x)
    sum2 = sum(math.cos(c * xi) for xi in x)
    term1 = -a * math.exp(-b * math.sqrt(sum1 / d))
    term2 = -math.exp(sum2 / d)
    return term1 + term2 + a + math.e


def rastrigin_function(x):
    d = len(x)
    A = 10
    sum1 = sum(xi ** 2 - A * math.cos(2 * math.pi * xi) for xi in x)
    return A * d + sum1


def schwefel_function(x):
    d = len(x)
    sum1 = sum(xi * math.sin(math.sqrt(abs(xi))) for xi in x)
    return 418.9829 * d - sum1


def rosenbrock_function(x):
    d = len(x)
    sum1 = sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(d - 1))
    return sum1



# --- Algoritmo Genético Real-Codificado com GRÁFICO ---

def genetic_algorithm(objective_func, n_dims, pop_size, generations, bounds, mutation_rate=0.1, elite_size=1, func_name=""):
    def create_individual():
        return np.random.uniform(bounds[:, 0], bounds[:, 1], n_dims)

    def crossover(parent1, parent2):
        alpha = np.random.rand()
        return alpha * parent1 + (1 - alpha) * parent2

    def mutate(individual):
        mutation = np.random.normal(0, 0.1, n_dims)
        mutated = individual + mutation
        return np.clip(mutated, bounds[:, 0], bounds[:, 1])

    def tournament_selection(pop, fitnesses, k=3):
        indices = np.random.choice(len(pop), k)
        selected = min(indices, key=lambda i: fitnesses[i])
        return pop[selected]

    population = [create_individual() for _ in range(pop_size)]
    best_solution = None
    best_fitness = float('inf')
    fitness_history = []  # 📈 histórico do melhor fitness

    for gen in range(generations):
        fitnesses = [objective_func(ind) for ind in population]
        min_idx = np.argmin(fitnesses)
        current_best = fitnesses[min_idx]

        if current_best < best_fitness:
            best_fitness = current_best
            best_solution = population[min_idx]

        fitness_history.append(best_fitness)

        elites = [population[i] for i in np.argsort(fitnesses)[:elite_size]]
        new_population = elites.copy()

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child)
            new_population.append(child)

        population = new_population

    # 📊 Plota o gráfico após a execução
    plt.figure()
    plt.plot(fitness_history, label="Fitness mínimo")
    plt.title(f"Convergência - {func_name}")
    plt.xlabel("Geração")
    plt.ylabel("Melhor Fitness")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"convergencia_{func_name.lower()}.png") # Salva o gráfico como PNG
    # --------------------------

    plt.show()

    return best_solution, best_fitness


# --- Benchmark com gráfico para cada função ---

def run_benchmark_ga():
    n_dims = 30
    pop_size = 30
    generations = 100

    benchmarks = [
        ("Ackley", ackley_function, np.array([[-30, 30]] * n_dims)),
        ("Rastrigin", rastrigin_function, np.array([[-5.12, 5.12]] * n_dims)),
        ("Schwefel", schwefel_function, np.array([[-500, 500]] * n_dims)),
        ("Rosenbrock", rosenbrock_function, np.array([[-5, 10]] * n_dims)),
    ]

    for name, func, bounds in benchmarks:
        print(f"\n🏁 Otimizando função {name}")
        best_solution, best_fitness = genetic_algorithm(
            objective_func=func,
            n_dims=n_dims,
            pop_size=pop_size,
            generations=generations,
            bounds=bounds,
            mutation_rate=0.2,
            elite_size=2,
            func_name=name  # usado no gráfico
        )
        print(f"✅ Resultado {name}:")
        print(f"  Melhor solução: {best_solution}")
        print(f"  Fitness: {best_fitness:.10f}")


if __name__ == '__main__':
    run_benchmark_ga()
