from deap import tools
from deap import algorithms


def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """
    Evolutionary Algorithm with Elitism.

    This algorithm is similar to DEAP's `eaSimple()` algorithm, with the modification that
    `halloffame` is used to implement an elitism mechanism. The individuals contained in
    the `halloffame` are directly injected into the next generation and are not subject
    to the genetic operators of selection, crossover, and mutation.

    Args:
        population (list): The initial population of individuals.
        toolbox (deap.base.Toolbox): The DEAP toolbox with operators.
        cxpb (float): Crossover probability (0.0 to 1.0).
        mutpb (float): Mutation probability (0.0 to 1.0).
        ngen (int): The number of generations to run the algorithm.
        stats (deap.tools.Statistics): Statistics object to collect data (optional).
        halloffame (deap.tools.HallOfFame): Hall of Fame to store best individuals.
        verbose (bool): Whether to print verbose output (default is debug setting).

    Returns:
        list: The final population of individuals after evolution.
        deap.tools.Logbook: Logbook containing evolution statistics.

    Raises:
        ValueError: If `halloffame` is empty.

    Note:
    - The `population` list is modified in place to store the final generation.
    - The `halloffame` is used to store the best individuals and is updated during evolution.
    - The `stats` object can be used to collect and report statistics during evolution.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the best back to population:
        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

