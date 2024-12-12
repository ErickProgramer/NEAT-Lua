--- Evolve your @{Genome} networks!
-- The NEAT algorithm evolves based on generations
-- so, this class allows you to have a Population that
-- evolves along the time by doing crossover, mutations etc.
-- But it's up to you define your fitness logic so the algorithm can evolve.
-- @module Population
-- @usage
-- local Population = require "NEAT.Population"

local success, Genome = pcall(require, "NEAT.Genome")

if not success then
    Genome = require "Genome"
end

-- importing Genome functions into locals for performance.
local getCompatibility = Genome.getCompatibility
local genome_crossover = Genome.crossover
local genome_mutate = Genome.mutate
local genome_forward = Genome.forward
local genome_isGenome = Genome.isGenome

local function sigmoid(x)
    return 1/(1 + math.exp(-x))
end

-- importing Lua's standard functions
local tsort = table.sort
local floor = math.floor
local random = math.random
local mceil = math.ceil
local assert = assert
local fmt = string.format
local tconcat = table.concat
local print = print
local setmetatable = setmetatable

local success, clear = pcall(require, "table.clear")
if not success then -- lets create our own
    function clear(t)
        for i=1, #t do
            t[i] = nil
        end
    end
end

local function orderByFitness(a1, a2)
    return a1.fitness < a2.fitness
end

local Population_mutate, Population_crossover, Population_adjustFitness, Population_speciate, Population_killHalf, Population_evolve, Population_getBetterFitness, Population_getWorstFitness, Population_setCompatibilityThresold, Population_removeAgent
local Population = {}
Population.__index = Population

--- Functions
-- @section Functions

--- Creates a new @{Population} object
-- @usage
-- local population = Population.new(3, 4) -- each agent in this Population must have 3 input nodes and 4 output nodes
-- @param n_inputs an integer representing the required number of input nodes for each Genome
-- @param n_outputs an integer representing the required number of output nodes for each Genome
function Population.new(n_inputs, n_outputs)
    local self = setmetatable({
        n_inputs = n_inputs,
        n_outputs = n_outputs,
        population = {},
        species = {},
    }, Population)

    Population_setCompatibilityThresold(self, 0.5)

    return self
end

--- Methods
-- @section Methods

--- Sets the maximum value to accept a @{Genome} in a species (it's 0.5 by default)
-- @param thresold an integer
function Population:setCompatibilityThresold(thresold)
    self._compatibility_thresold = thresold
end

--- Adds a certain quantity of agents to the population
-- @param size an integer representing how much agents will be inserted
function Population:addAgents(size)
    local n_inputs = self.n_inputs
    local n_outputs = self.n_outputs
    self.cache = nil
    local population = self.population
    for i=1, size do
        local a = Genome.new(n_inputs, n_outputs)
        local idx = #population+1
        population[idx] = a
        population[a] = idx
    end
    self.size = #population
end

--- Adds an agent to the population
-- @param agent a @{Genome} object
function Population:insertAgent(agent)
    agent.activations[1] = sigmoid
    assert(genome_isGenome(agent), "'agent' is not a genome")
    assert(agent.n_inputs == self.n_inputs and agent.n_outputs == self.n_outputs,
           fmt("'agent' is not compatible with this population (input expected: {%d,%d} got {%d,%d}", self.n_inputs or 0, self.n_outputs or 0, agent.n_inputs or 0, agent.n_outputs or 0))

    local population = self.population
    local idx = #population+1
    population[idx] = agent
    population[agent] = idx
end

--- Removes an agent from the population
-- @usage
-- local pop = Population.new(4, 3)
--
-- local g = Genome.new(4, 3)
-- pop:insertAgent(g)
-- print(#pop.population) -- 1
-- pop:removeAgent(g)
-- print(#pop.population) -- 0
-- pop:
-- @param agent a @{Genome} object to be removed
function Population:removeAgent(agent)
    local population = self.population

    local idx = population[agent]
    if not idx then return end
    local tmp = population[#population]

    population[idx] = tmp
    population[tmp] = idx
    population[#population] = nil
end

--- Makes the mutation process
-- @param n an integer representing how many agents will mutate
function Population:mutate(n)
    local population = self.population
    local n_pop = #population
    if n_pop == 0 then return end
    for i=1, n do
        local ind = population[random(n_pop)]
        genome_mutate(ind)
    end
end

--- Makes the speciation process (split agents into species)
function Population:speciate()
    local species = self.species
    clear(species) -- reseting the species

    local population = self.population
    local cmp_thresold = self._compatibility_thresold
    local n = 0

    for i=1, #population do
        local genome = population[i]
        local inserted = false
        n = #species

        for j=1, n do
            local spec = species[j]

            local representative = getCompatibility(spec.representative, genome)
            if representative <= cmp_thresold then
                genome:setSpeciesBelongs(spec)
                spec[#spec+1] = genome
                inserted = true
                break
            end
        end

        if not inserted then -- create a new species with an unique agent
            local new =  { genome, representative = genome, id=#species+1 }
            genome:setSpeciesBelongs(new)
            species[#species+1] = new
        end
    end
end

--- Currently not used
function Population:killHalf()
    local species = self.species
    for i=1, #species do
       local spec = species[i]
       local thresold = mceil(#spec * 0.5)
       for j=thresold, #spec do
            Population_removeAgent(self, spec[j])
       end
    end
end

--- Adjust the fitness of every entity
-- The equation to adjust the fitness of an individue is:
-- new_fitness = old_fitness / N
-- where N is the number of members of the species the indivue belongs to.
function Population:adjustFitness()
    local species = self.species
    for i=1, #species do
        local spec = species[i]
        local n = #spec
        for j=1, n do
            local member = spec[j]
            member.fitness = member.fitness / n
        end
    end
end

--- Makes the crossover process
-- This method iterate over every species and keeps the better individue
-- in the next generation, and if the species has more than 1 individue,
-- it makes the crossover over the better individue and the second better individue
function Population:crossover()
    self.cache = nil
    local species = self.species

    if #species == 0 then return end

    local new_population = self.population
    clear(new_population)

    for i=1, #species do
        local spec = species[i]

        tsort(spec, orderByFitness)
        local better = spec[#spec]

        new_population[#new_population+1] = better
        better.fitness = 0

        if #spec >= 2 then
            local second_better = spec[#spec-1]
            for j=1, #spec-1 do
                local new = genome_crossover(better, second_better)
                new_population[#new_population+1] = new
            end
        end

    end
end

--- Makes the population evolve by calling @{adjustFitness}, @{crossover}, @{mutate} & @{speciate}
-- @param[opt] n_mut an integer to be passed to @{mutate}
function Population:evolve(n_mut)
    Population_adjustFitness(self)
    Population_crossover(self)
    Population_mutate(self, n_mut or 2)
    Population_speciate(self)
end

-- Gets the worst fitness from population
-- @return a number representing the fitness
-- @return which agent is
function Population:getWorstFitness()
    local population = self.population
    tsort(population, orderByFitness)
    local worst_agent = population[1]

    return worst_agent.fitness, worst_agent
end

-- Gets the better fitness from population
-- @return a number representing the fitness
-- @return which agent is
function Population:getBetterFitness()
    local population = self.population
    tsort(population, orderByFitness)

    local better_agent = population[#population]
    return better_agent.fitness, better_agent
end

--- Run a population for N generations
-- You can use it to train the AIs in a specific dataset,
-- but it's not useful for games for example.
--
-- @usage
-- local Population = require("Population")
-- 
-- -- Learning the OR operation
-- local p = Population.new(2, 1)
-- 
-- p:setCompatibilityThresold(0.2)
-- 
-- -- Creating an initial population with 100 agents
-- p:addAgents(100)
-- 
-- local dataset = {
--     { {0,0}, {0} },
--     { {0,1}, {1} },
--     { {1,0}, {1} },
--     { {1,1}, {1} }
-- }
-- 
-- local function fitness_eval(population, agent, state, response, expected)
--     local dist = math.abs(expected[1] - response[1])
--     agent.fitness = agent.fitness + (1 - dist)  -- Max fitness is 1, minimize dist
-- end
-- 
-- local n_epochs = 1000
-- local debug_step = 100
-- local better = p:run(fitness_eval, dataset, n_epochs, debug_step)
-- 
-- -- if you have binser installed, you can save this agent in a file:
-- better:save("myagent.save")
--
-- @param responseObserver the fitness function, use it to make your fitness logic.
-- @param dataset the dataset table
-- @param generations the number of generations to run
-- @param debug_steps each `debug_steps` generation, it'll show debug informations
-- @param n_mut How many entities will be mutate each new generation
-- @return a @{Genome} representing the better agent from the last generation
function Population:run(responseObserver, dataset, generations, debug_steps, n_mut)
    debug_steps = debug_steps or floor(generations * 0.1)
    n_mut = n_mut or 10
    if not responseObserver then
        error("responseObserver is required")
    end

    for i=1, generations do
        local population = self.population

        for j=1, #population do
            local agent = population[j]
            for k=1, #dataset do
                local batch = dataset[k]
                local state = batch[1]
                local expected = batch[2]
                local ind = population[j]
                local resp = genome_forward(agent, state)
                responseObserver(self, ind, state, resp, expected)
            end
        end
        Population_evolve(self, n_mut)
        if i % debug_steps == 0 or i == 1 then
            local better_fitness = Population_getBetterFitness(self)
            local worst_fitness = Population_getWorstFitness(self)
            print("Generation:     "..i)
            print("Better fitness: "..better_fitness)
            print("Worst fitness:  "..worst_fitness)
            print("Species:        " .. #self.species)
            print("Population:     "..#population)
            print("----------------")
        end
    end

    print("Running the better agent...")
    local _, better = Population_getBetterFitness(self)
    for i=1, #dataset do
        local batch = dataset[i]
        local x = batch[1]
        print("Forward: " .. tconcat(x, ", ") .. " -> " .. tconcat(genome_forward(better, x)))
    end

    return better
end

Population_mutate = Population.mutate
Population_speciate = Population.speciate
Population_crossover = Population.crossover
Population_adjustFitness = Population.adjustFitness
Population_evolve = Population.evolve
Population_getBetterFitness = Population.getBetterFitness
Population_getWorstFitness = Population.getWorstFitness
Population_setCompatibilityThresold = Population.setCompatibilityThresold
Population_removeAgent = Population.removeAgent

return Population
