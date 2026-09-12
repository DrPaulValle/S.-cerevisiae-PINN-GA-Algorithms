function result = run_genetic_algorithm(objective, bounds, pop_size, generations, elite_frac, crossover_rate, mutation_rate, mutation_scale, tournament_size, stall_generations, seed, blx_alpha, verbose)
    stream = RandStream('mt19937ar', 'Seed', seed);
    n_vars = size(bounds, 1);
    lo = bounds(:,1).';
    hi = bounds(:,2).';

    pop = rand(stream, pop_size, n_vars) .* (hi - lo) + lo;
    fitness = zeros(pop_size, 1);
    for i = 1:pop_size
        fitness(i) = objective(pop(i,:));
    end

    [best_fit, best_idx] = min(fitness);
    best = pop(best_idx, :);
    history = best_fit;
    no_improve = 0;
    n_elite = max(1, round(elite_frac * pop_size));

    for gen = 1:generations
        [fitness, order] = sort(fitness, 'ascend');
        pop = pop(order, :);

        if fitness(1) < best_fit
            best_fit = fitness(1);
            best = pop(1, :);
            no_improve = 0;
        else
            no_improve = no_improve + 1;
        end

        history(end+1,1) = best_fit; %#ok<AGROW>

        if verbose && (mod(gen-1, 500) == 0 || gen == generations)
            fprintf('      GA gen %4d | best objective = %.6e\n', gen-1, best_fit);
        end

        if no_improve >= stall_generations
            if verbose
                fprintf('      Early stop GA at generation %d (stall_generations=%d)\n', gen-1, stall_generations);
            end
            break;
        end

        new_pop = pop(1:n_elite, :);

        while size(new_pop, 1) < pop_size
            parent1 = tournament_select(pop, fitness, stream, tournament_size);
            parent2 = tournament_select(pop, fitness, stream, tournament_size);

            if rand(stream) < crossover_rate
                [child1, child2] = blend_crossover(parent1, parent2, stream, blx_alpha);
            else
                child1 = parent1;
                child2 = parent2;
            end

            progress = (gen - 1) / max(generations - 1, 1);
            mut_scale_now = mutation_scale * (1.0 - 0.60 * progress);

            child1 = gaussian_mutation(child1, bounds, stream, mutation_rate, mut_scale_now);
            child2 = gaussian_mutation(child2, bounds, stream, mutation_rate, mut_scale_now);

            new_pop = [new_pop; child1]; %#ok<AGROW>
            if size(new_pop, 1) < pop_size
                new_pop = [new_pop; child2]; %#ok<AGROW>
            end
        end

        pop = new_pop;
        fitness = zeros(pop_size, 1);
        for i = 1:pop_size
            fitness(i) = objective(pop(i,:));
        end
    end

    result.Best = best;
    result.fun = best_fit;
    result.history = history;
    result.n_generations = numel(history) - 1;
    result.seed = seed;
    result.pop_size = pop_size;

    function selected = tournament_select(pop, fitness, stream, tournament_size)
        idx = randperm(stream, size(pop,1), tournament_size);
        [~, loc] = min(fitness(idx));
        selected = pop(idx(loc), :);
    end
    
    function [c1, c2] = blend_crossover(p1, p2, stream, alpha)
        gamma = -alpha + (1.0 + 2.0*alpha) * rand(stream, 1, numel(p1));
        c1 = gamma .* p1 + (1.0 - gamma) .* p2;
        c2 = gamma .* p2 + (1.0 - gamma) .* p1;
    end
    
    function out = gaussian_mutation(child, bnd, stream, rate, scale)
        out = child;

        for jj = 1:size(bnd,1)
            lo_j = bnd(jj,1);
            hi_j = bnd(jj,2);

            if rand(stream) < rate
                sigma = scale*(hi_j-lo_j);
                out(jj) = out(jj) + sigma*randn(stream);
            end

            % Small probability of global reinitialization of one gene.
            if rand(stream) < 0.02
                out(jj) = lo_j + (hi_j-lo_j)*rand(stream);
            end

            out(jj) = min(max(out(jj),lo_j),hi_j);
        end
    end

end