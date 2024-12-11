--- Create & manipulate genomes
-- @usage
-- local Genome = require "NEAT.Genome"
-- @module Genome

local random = math.random
local fmt = string.format
local assert = assert
local error = error
local tsort = table.sort
local mmax = math.max
local mmin = math.min
local mabs = math.abs
local setmetatable = setmetatable
local type = type
local rawget = rawget
local select = select
local pairs = pairs

local package = package
local old_path = package.path
local function searchIn(...)
    for i=1, select("#", ...) do
        local path = select(i, ...)
        package.path = package.path .. ";" .. path
    end
end

searchIn("?.lua",
         "?/init.lua",
         "?/?.lua",
         "libs/?.lua",
         "libs/?/init.lua",
         "libs/?/?.lua")

local binser_support, binser = pcall(require, "binser")
local binser_writeFile, binser_readFile

if binser_support then
    binser_writeFile = binser.writeFile
    binser_readFile = binser.readFile
end

package.path = old_path

local function orderConnections(c1, c2)
    return c1.innov > c2.innov
end

local _neatKey = {}

-- PERFORMANCE TIP: methods into locals
local Genome_new, Genome_newNode, Genome_getNode, Genome_activate, Genome_existConnection, Genome_getExcess, Genome_mutateWeight, Genome_resetOutputs, Genome_newConnection, Genome_getConnection, Genome_newNodes, Genome_mutateNewConnection, Genome_getAvarageWeightDiff, Genome_setupMutateProbs, Genome_mutateChangeActivation, Genome_mutateSplitConnection, Genome_getMatchingConnections, Genome_getDisjointConnections, Genome_existInFreeConnections, Genome_setAsFreeConnection, Genome_setEveryConnectionAsFree

local Genome = {}
Genome.__index = Genome

--- Types
-- @section Types

------
-- The node object.
-- @field out number The output of the node
-- @field kind A string representing the kind of the node. Can be: input, hidden, output
-- @field f function
-- @table Node

------
-- The connection object.
-- @field weight a number representing the weight of the connection
-- @field in_node a @{Node} representing the input node
-- @field out_node a @{Node} representing the output node
-- @field innov an integer representing the [innovation](https://youtube.com) number
-- @table Connection

--- Functions
-- @section Functions

--- Creates a new Genome object
-- @function Genome:new
-- @param n_inputs  integer
-- @param n_outputs integer
function Genome.new(n_inputs, n_outputs)
    assert(n_inputs and n_outputs, "missing n_inputs or n_outputs")

    local self = setmetatable({
        [_neatKey]=true,
        n_inputs = n_inputs,
        n_outputs = n_outputs,
        nodes = {
            input = {},
            hidden = {},
            output = {},
            id=0
        },

        connections = {
            innov = 0,

            exist_connection = {},
        },

        free_connections = {
            -- input -> hidden
            input_hidden = {},

            -- input -> output
            input_output = {},

            -- hidden -> hidden
            hidden_hidden = {},

            -- output -> hidden
            hidden_output = {},
        },

        innov_connection = {},

        activations = {},

        fitness = 0,
    }, Genome)

    Genome_newNodes(self, "input", n_inputs)
    Genome_newNodes(self, "output", n_outputs)
    self.lockInputOutput_nodes = true

    Genome_setupMutateProbs(self)

    return self
end

function Genome.setRandomGenerator(f)
    random = f
end

function Genome.getRandomGenerator()
    return random
end



--- Returns true if `obj` is a Genome instance
-- @usage
-- print(Genome.isGenome(Genome)) -- false because it's not an instance
-- print(Genome.isGenome({})) -- false
-- print(Genome.isGenome(Genome.new(2,2)) -- true
-- print(Genome.isGenome("")) -- false
---@param obj any
function Genome.isGenome(obj)
    return type(obj) == "table" and (rawget(obj, _neatKey) == true)
end

--- Methods
-- @section Methods

do
    local function copy_node(node)
        return { kind = node.kind, f = node.f, out = node.out, id = node.id }
    end

    --- Returns an identical copy of the @{Genome}
    -- @usage
    -- local mynn = Genome.new(5,5)
    --
    -- local copy = mynn:copy()
    --
    -- print(mynn)
    -- print(copy)
    -- @return @{Genome}
    function Genome:copy()
        local res = Genome_new(self.n_inputs, self.n_outputs)
        local rnodes = res.nodes

        do -- copying nodes
            local nodes = self.nodes

            rnodes.id = nodes.id
            for i=1, #nodes do
                local node = nodes[i]
                local nodes_subset = nodes[node.kind]
                local cp = copy_node(node)
                nodes_subset[#nodes_subset+1] = cp
                rnodes[i] = cp
            end
        end

        do -- copying connections
            local rconnections = res.connections
            local connections = self.connections
            local innov_connection = res.innov_connection

            local exist_connection = rconnections.exist_connection
            for k, v in pairs(connections.exist_connection) do
                exist_connection[k] = v
            end

            rconnections.innov = connections.innov
            for i=1, #connections do
                local connection = connections[i]
                local innov = connection.innov

                local rconnection = {
                    weight = connection.weight,
                    in_node = rnodes[connection.in_node.id],
                    out_node = rnodes[connection.out_node.id],
                    innov = innov
                }

                innov_connection[innov] = rconnection

                rconnections[i] = rconnection
            end
        end
        return res
   end
end

--- Sets every connection related to a node as free.
-- This method is for internal usage, it's not recommended to use it at all
function Genome:setEveryConnectionAsFree(a1, a2)
    if type(a1) == "string" then
        local other_kind, node = a1, a2
        local nodes = self.nodes[other_kind]

        for i=1, #nodes do
            local knode = nodes[i]
            local connection = { in_node = knode, out_node = node }
            if (not Genome_existInFreeConnections(self, connection)) and knode ~= node then
                Genome_setAsFreeConnection(self, connection)
            end
        end
    elseif type(a2) == "string" then
        local other_kind, node = a2, a1
        local nodes = self.nodes[other_kind]

        for i=1, #nodes do
            local knode = nodes[i]
            local connection = { in_node = node, out_node = knode }
            if (not Genome_existInFreeConnections(self, connection)) and knode ~= node then
                Genome_setAsFreeConnection(self, connection)
            end
        end
    end
end

--- Creates a new node in the class.
-- Generally you can only add hidden nodes,
-- because the creation of input/output nodes are locked
-- once you create an instance of @{Genome}
-- @usage
-- local mynn = Genome.new(2, 3)
-- mynn:newNode "hidden"
-- mynn:newNode "input" -- error
-- @param kind string
-- @param activation function
-- @return @{Node}
function Genome:newNode(kind, activation)
    if self.lockInputOutput_nodes and (kind == "input" or kind == "output") then
        error("cannot create more input/output nodes.")
    end

    local nodes = self.nodes

    if #nodes[kind] > 0 and nodes[#nodes].kind ~= kind then
        error("the nodes order is wrong.")
    end

    nodes.id = nodes.id + 1
    local node_id = nodes.id

    local activations = self.activations
    local node = {
        kind = kind,
        id = node_id,
        out = 0,
        f = activation or activations[random(#activations)]
    }

    local arr = nodes[kind]
    arr[#arr+1] = node

    nodes[#nodes+1] = node

    if kind == "input" then
        Genome_setEveryConnectionAsFree(self, node, "output")
    elseif kind == "output" then
        Genome_setEveryConnectionAsFree(self, "input", node)
        Genome_setEveryConnectionAsFree(self, "hidden", node)
    elseif kind == "hidden" then -- making input -> hidden free connections
        Genome_setEveryConnectionAsFree(self, "input", node)
        Genome_setEveryConnectionAsFree(self, node, "output")
        Genome_setEveryConnectionAsFree(self, "hidden", node)
    end

    return node
end

--- Create multime nodes _amount_ times.
--@param kind The kind of the node
--@param amount How much nodes to be inserted
--@param activation The activation function of the node
function Genome:newNodes(kind, amount, activation)
    for i=1, amount do
        Genome_newNode(self, kind, activation)
    end
end

--- Returns true if the connection exist
-- @param in_node The input node. Can be an integer or a @{Node}
-- @param out_node The output node. Can be an integer or a @{Node}
-- @return true or false
function Genome:existConnection(in_node, out_node)
    in_node = type(in_node) == "number" and in_node or in_node.id
    out_node = type(out_node) == "number" and out_node or out_node.id

    local exist_connection = self.connections.exist_connection
    return exist_connection[in_node.."-"..out_node] or
           exist_connection[out_node.."-"..in_node] or
           in_node == out_node
end

--- Sets the given connection as free
-- This function is for internal usage.
-- @param connection @{Connection}
function Genome:setAsFreeConnection(connection)
    local in_node = connection.in_node
    local out_node = connection.out_node
    local in_kind = in_node.kind
    local out_kind = out_node.kind

    local free_connections = self.free_connections
    local free = free_connections[in_kind.."_"..out_kind]
    if not free then
        error("Cannot set this conection as free")
    end

    local key = in_node.id.."-"..out_node.id
    local connection = { in_node.id, out_node.id }

    do
        local idx = #free+1
        free[idx] = connection
        free[key] = idx
    end

    do
        local idx = #free_connections+1
        free_connections[idx] = connection
        free_connections[key] = idx
    end
end

--- Return a random free connection
-- @return An array of size 2 representing the connection. E.g: {2,3}
function Genome:pickRandomFreeConnection()
    local fc = self.free_connections
    return fc[random(#fc)]
end

--- Returns true if the given connection is free
-- @param connection a @{Connection} object. But it's only required the in_node and out_node field.
-- @return boolean
function Genome:existInFreeConnections(connection)
    local fc = self.free_connections
    return fc[connection.in_node.id.."-"..connection.out_node.id] ~= nil
end

--- Removes a connection from the free connections
function Genome:removeFromFreeConnections(in_node, out_node)
    in_node = Genome_getNode(self, in_node)
    out_node = Genome_getNode(self, out_node)

    local in_kind = in_node.kind
    local out_kind = out_node.kind

    local free_connections = self.free_connections
    local free = free_connections[in_kind.."_"..out_kind]
    if not free then
        return
    end

    local key = in_node.id.."-"..out_node.id
    do
        local idx = free[key]

        local pos = free[idx]
        free[idx] = nil

        local tmp = free[#free]

        if tmp then
            free[pos] = tmp
            free[tmp[1].."-"..tmp[2]] = pos
            free[#free] = nil
        end

        free[key] = nil
    end

    do
        local idx = #free_connections+1

        local pos = free_connections[key]
        free_connections[idx] = nil

        local tmp = free_connections[#free_connections]
        if tmp then
            free_connections[pos] = tmp
            free_connections[tmp[1].."-"..tmp[2]] = pos
            free_connections[#free_connections] = nil
        end

        free_connections[key] = nil
    end
end

--- Returns a connection by innovation number
function Genome:getConnection(innov)
    return self.innov_connection[innov]
end

-- Return a node by its ID
function Genome:getNode(id)
    return self.nodes[id]
end

--- Changes an existing connection.
function Genome:setConnection(in_node, out_node, weight, enabled, innov)
    local exist_connection = self.connections.exist_connection
    local connection = exist_connection[in_node.."-"..out_node] or
                       exist_connection[out_node.."-"..in_node]

    connection.in_node = Genome_getNode(self, in_node)
    connection.out_node = Genome_getNode(self, out_node)
    connection.weight = weight
    connection.enabled = enabled
    connection.innov = innov
end

--- Creates a new connection
-- @usage
-- local mynn = Gnome.new(1, 1)
-- mynn:newConnection(1, 2, 0.5)
-- print(mynn:forward({ 5 }))
-- @return Connection
function Genome:newConnection(in_node, out_node, weight, enabled, innov)
    in_node = type(in_node) == "number" and Genome_getNode(self, in_node) or in_node
    out_node = type(out_node) == "number" and Genome_getNode(self, out_node) or out_node

    if Genome_existConnection(self, in_node, out_node) then
        error("connection " .. in_node.id .. "-" .. out_node.id .." already exists")
    end

    local connections = self.connections
    local connection = {
        weight = weight or random(),
        in_node = in_node,
        out_node = out_node,
        innov = innov or connections.innov+1,
        enabled = enabled or enabled == nil,
    }
    connection.out_node.connecting = connection

    local cinnov = connection.innov
    if cinnov > connections.innov then
        connections.innov = cinnov
    end
    in_node.connection = connection


    self.innov_connection[connection.innov] = connection

    connections[#connections+1] = connection
    connections.exist_connection[in_node.id .. "-" .. out_node.id] = connection

    if self:existInFreeConnections(connection) then
        self:removeFromFreeConnections(in_node.id, out_node.id)
    end

    if self:existInFreeConnections({ in_node=out_node, out_node=in_node }) then
        self:removeFromFreeConnections(out_node.id, in_node.id)
    end

    return connection
end

--- Resets the outputs of each node to zero.
function Genome:resetOutputs()
    local nodes = self.nodes
    for i=1, #nodes do
        local node = nodes[i]
        if node.kind ~= "input" then
            nodes[i].out = 0
        end
    end
end

--- Activates each node by calling its activation function.
function Genome:activate()
    local nodes = self.nodes
    for i=1, #nodes do
        local node = nodes[i]
        local f = node.f
        if f then
            node.out = f(node.out)
        end
    end
end

--- Setups the probabilities for each mutation happens when call the `:mutate` method.
-- If you don't call this method the default probabilities will be setted.
-- @param info table
-- @usage
-- local mynn = Genome.new(3, 1)
--
-- -- By default, the configurations are:
-- mynn:setupMutateProbs{
--  activationChange = 0.1, -- 10% chance to an activation function be changed
--  splitConnection = 0.2, -- 20% chance to a connection be splitted
--  weightChange = 0.3, -- 50% chance to a weight value be changed
--  newConnection = 0.5, -- 10% chance to a connection be created
--  newNode = 0.1 -- 10% chance to a new node be created
-- }
-- @see mutate
function Genome:setupMutateProbs(info)
    info = info or {}
    self._mutateChangeActivation = info.activationChange or 0.1
    self._mutateSplitConnection = info.splitConnection or 0.2
    self._mutateWeight = info.weightChange or 0.3
    self._mutateNewConnection = info.newConnection or 0.5
    self._mutateNewNode = info.newNode or 0.1
    return self
end

--- Adds a new activation function that can be used by the class
-- @see setupMutateProbs
-- @see mutate
function Genome:addActivationFunction(f)
    local activations = self.activations
    activations[#activations+1] = f
end

--- Choose a random node and change its activation function
-- @see setupMutateProbs
-- @see mutate
function Genome:mutateChangeActivation()
    local nodes = self.nodes
    local activations = self.activations

    local node = nodes[random(#nodes)]

    node.f = activations[random(#activations)]

    return node
end

---
-- Slipts a connection, if not passed, it chooses a random connection and split it.
-- @see setupMutateProbs
-- @see mutate
---@param connection[opt] table
function Genome:mutateSplitConnection(connection)
    local connections = self.connections
    connection = connection or connections[random(#connections)]
    if not connection then
        return
    end

    connection.enabled = false
    local node = Genome_newNode(self, "hidden")

    Genome_newConnection(self, connection.in_node.id, node.id, 1)
    Genome_newConnection(self, node.id, connection.out_node.id, connection.weight)

    return node
end

--- selects a random connection and change its weight
-- @usage
-- math.randomseed(69) -- for same results
-- local nn = Genome.new(3, 5)
--
-- local connection = nn:mutateNewConnection()
--
-- print(connection.weight) -- 0.40090842076939
-- nn:mutateWeight()
-- print(connection.weight) -- 0.28136715285445
---@return number | nil
function Genome:mutateWeight()
    local connections = self.connections
    local connection = connections[random(#connections)]
    if connection then
        local w = connection.weight + random() - 0.5
        connection.weight = w
        return w
    end
    return nil
end

--- Adds a new connection and returns if possible
-- @usage
-- local nn = Genome.new(3, 2)
-- print(#nn.connections) -- 0
-- nn:mutateNewConnection()
-- print(#nn.connections) -- 1
---@return table | nil
function Genome:mutateNewConnection()
    local co = self:pickRandomFreeConnection()

    -- can be nil if there's no free connections
    if co then
        local in_node, out_node = co[1], co[2]
        return Genome_newConnection(self, in_node, out_node)
    end

    return nil
end

--- Mutates the Network randomly based on the probabilities setted
-- @see setupMutateProbs
function Genome:mutate()
    if random() <= self._mutateNewNode then
        Genome_newNode(self, "hidden")
    end

    if random() <= self._mutateChangeActivation then
        Genome_mutateChangeActivation(self)
    end

    if random() <= self._mutateWeight then
        Genome_mutateWeight(self)
    end

    if random() <= self._mutateNewConnection then
        Genome_mutateNewConnection(self)
    end

    if random() <= self._mutateSplitConnection then
        Genome_mutateSplitConnection(self)
    end
end

--- Makes the forward process and returns a vector
-- @usage
-- math.randomseed(69) -- for same results
-- local nn = Genome.new(3, 4) -- receives 3 inputs and produces 4 numbers
-- 
-- -- Making a fully [connected network](https://youtube.com)
-- for i, in_node in ipairs(nn.nodes.input) do
--     for j, out_node in ipairs(nn.nodes.output) do
--         nn:newConnection(in_node, out_node)
--     end
-- end
--  -- We can send input only for selected input nodes.
--  -- In this case, only the 2nd input node will receive a value
--  -- the rest will be setted as 0
-- local out = nn:forward({ [2] = 3 })
-- print(table.concat(out, ", ")) -- 1.1413761962552, 1.7518015494614, 2.1827942531713, 1.1897555017687
---@param input table
function Genome:forward(input)
    Genome_resetOutputs(self)

    local nodes = self.nodes
    if input then
        local input_nodes = nodes.input

        for i=1, #input_nodes do
            local value = input[i] or 0
            input_nodes[i].out = value
        end
    end

    local connections = self.connections
    for i=1, #connections do
        local connection = connections[i]
        if connection.enabled then
            local in_node = connection.in_node
            local out_node = connection.out_node
            out_node.out = out_node.out + in_node.out*connection.weight
        end
    end
    Genome_activate(self)

    local output = {}
    local output_nodes = nodes.output
    for i=1, #output_nodes do
        output[i] = output_nodes[i].out
    end

    return output
end

--- Returns an integer number which represents what action the AI did.
-- it's like the @{forward} method but instead of return a vector with the outputs, it returns the "action" of the AI
-- based on the maximum value. It's useful when you want an AI to play a game and make it do actions
-- @usage
-- local mynn = Genome.new(3, 5) -- receives 3 inputs, and can do 5 actions
-- local JUMP = 1
-- local WALK_LEFT = 2
-- local WALK_RIGHT = 3
-- local WALK_STOP = 4
-- local ATTACK = 5
--
-- local action = mynn:takeAction(Game.getState())
-- if action == JUMP then
--  Game.jump()
-- elseif action == WALK_LEFT then
--  Game.walkLeft()
-- elseif action == WALK_RIGHT then
--  Game.walkRight()
-- elseif action == WALK_STOP then
--  Game.walkStop()
-- elseif action == ATTACK then
--  Game.attack()
-- end
-- @see forward
---@param inp table
function Genome:takeAction(inp)
    local res = self:forward(inp)
    local max, idx = res[1], 1
    for i=2, #res do
        local v = res[i]
        if v > max then
            max = v
            idx = i
        end
    end
    return idx
end

--- Returns the disjoint connections
-- @return table
function Genome:getDisjointConnections(nn)
    local s_connections = self.connections
    local nn_connections = nn.connections

    tsort(s_connections, orderConnections)
    tsort(nn_connections, orderConnections)

    local m1, m2 = s_connections[1], nn_connections[1]
    if m1 then
        m1 = m1.innov
    else
        m1 = 1
    end
    if m2 then
        m2 = m2.innov
    else
        m2 = 1
    end

    local disjoint_c = {}
    local max_innov = m1 > m2 and m1 or m2
    local sfit = self.fitness > nn.fitness
    local sn_fit = not sfit

    for innov=1, max_innov do
        local e1 = Genome_getConnection(self, innov)
        local e2 = Genome_getConnection(self, innov)

        if (e1 and (not e2)) and sfit then
            disjoint_c[#disjoint_c+1] = e1
        elseif (e2 and (not e1)) and sn_fit then
            disjoint_c[#disjoint_c+1] = e2
        end
    end

    return disjoint_c
end

--- Returns a list of matching genes
-- @return table
function Genome:getMatchingConnections(nn)
    local s_connections = self.connections
    local nn_connections = nn.connections

    tsort(s_connections, orderConnections)
    tsort(nn_connections, orderConnections)

    local m1, m2 = s_connections[1], nn_connections[1]
    if m1 then
        m1 = m1.innov
    else
        m1 = 1
    end
    if m2 then
        m2 = m2.innov
    else
        m2 = 1
    end


    local matching_c = {}
    local max_innov = m1 > m2 and m1 or m2
    for innov=1, max_innov do
        local e1 = Genome_getConnection(self, innov)
        local e2 = Genome_getConnection(self, innov)
        if e1 and e2 then
            matching_c[#matching_c+1] = {e1,e2}
        end
    end

    return matching_c
end

--- Returns the number of excess genes
-- @param nn @{Genome}
-- @return integer
function Genome:getExcess(nn)
    local sconnections = self.connections
    local nnconnections = nn.connections

    tsort(sconnections, orderConnections)
    tsort(nnconnections, orderConnections)


    local sc_1 = sconnections[1]
    local nc_2 = nnconnections[1]

    local innov1 = sc_1 and sc_1.innov or 1
    local innov2 = nc_2 and nc_2.innov or 1

    local excess = mabs(innov1 - innov2)
    return excess
end

--- Computes the weight difference between two Genomes
function Genome:getAvarageWeightDiff(nn)
    local W = 0
    local match = Genome_getMatchingConnections(self, nn)
    for i=1, #match do
        local connections = match[i]
        W = W + mabs(connections[1].weight - connections[2].weight)
    end

    local n_match = #match

    return n_match == 0 and 0 or W/n_match
end

--- Returns how different the networks are
-- @usage
-- local nn1 = Genome.new(5, 3)
-- local nn2 = Genome.new(5, 3)
--
-- print(nn1:getCompatibility(nn2)) -- prints 0. They are 100% equal!
--
-- -- Modifying the networks...
-- for i=1, 8 do
--  nn1:mutate()
--  nn2:mutate()
-- end
-- print(nn1:getCompatibility(nn2))
-- @param nn @{Genome}
-- @param[opt] c1 The weight 1
-- @param[opt] c2 The weight 2
-- @param[opt] c3 The weight 3
function Genome:getCompatibility(nn, c1, c2, c3)
    c1=c1 or 1
    c2=c2 or 1
    c3=c3 or 1

    local N = mmax(#self.connections, #nn.connections)
    local E = Genome_getExcess(self, nn)
    local D = #Genome_getDisjointConnections(self, nn)
    local W = Genome_getAvarageWeightDiff(self, nn)

    N = N == 0 and 1e-10 or N -- to avoid zero division

    return ((c1 * E) / N) + ((c2 * D) / N) + (c3 * W)
end


--- Makes the crossover between two Genomes generating a "child"
-- @usage
-- local parent1 = Genome.new(2, 3)
-- local parent2 = Genome.new(2, 3)
--
-- for i=1, 5 do
--  parent1:mutate()
--  parent2:mutate()
-- end
--
-- local child = parent1:crossover(parent2)
-- print(child:getCompatibility(parent1), child:getCompatibility(parent2))
-- @param nn @{Genome}
-- @return a new @{Genome}
function Genome:crossover(nn)
    local nn_nodes = nn.nodes
    local self_nodes = self.nodes

    local res = Genome_new(#nn_nodes.input, #nn_nodes.output)

    local nn_hidden = nn_nodes.hidden
    local self_hidden = self_nodes.hidden
    local n_hidden = mmax(#nn_hidden, #self_hidden)

    for i=1, n_hidden do
        local n1, n2 = self_hidden[i], nn_hidden[i]
        local selected = (random() <= 0.5 and n1 or n2) or (n1 or n2)

        Genome_newNode(res, "hidden", selected.f)
    end

    local disjoint = Genome_getDisjointConnections(self, nn)
    local matching = Genome_getMatchingConnections(self, nn)

    for i=1, #matching do
        local connections = matching[i]
        local co1 = connections[1]
        local co2 = connections[2]

        local selected = random() <= 0.5 and co1 or co2
        local in_node = selected.in_node.id
        local out_node = selected.out_node.id

        if not Genome_existConnection(res, in_node, out_node) then
            Genome_newConnection(res, in_node, out_node, selected.weight, selected.enabled, selected.innov)
        end
    end

    for i=1, #disjoint do
        local co = disjoint[i]

        local in_node = co.in_node.id
        local out_node = co.out_node.id
        if not Genome_existConnection(res, in_node, out_node) then
            Genome_newConnection(res, in_node, out_node, co.weight, co.enabled, co.innov)
        end
   end

    tsort(res.connections, orderConnections)

    return res
end

--- Returns which species this Genome belongs to
-- If you're not using the class @{Population} it will not work.
function Genome:getSpeciesBelongs()
    return self.belongs_to_species
end

--- Sets to which species this Genome is from.
function Genome:setSpeciesBelongs(s)
    self.belongs_to_species = s
end

function Genome:__tostring()
    return fmt("Genome: %p", self)
end


if not binser_support then
    local function __CANT_USE__()
        error("binser is required to use this function. Install it in: https://github.com/bakpakin/binser")
    end

    Genome.save = __CANT_USE__
    Genome.load = __CANT_USE__
else
    --- Saves the Genome in the provided file
    -- This method only works if you have binser installed
    -- @param file a string representing the path
    function Genome:save(file)
        local objs = {}

        -- storing only data for performance.
        for k, v in pairs(self) do
            objs[k] = v
        end

        binser_writeFile(file, objs)
    end

    --- Loads a Genome network from the provided file
    -- @param file A string representing the path to the file
    -- @return @{Genome}
    function Genome.load(file)
        local from_file = binser_readFile(file)[1]
        local obj = setmetatable({}, Genome)

        for k, v in pairs(from_file) do
            obj[k] = v
        end
        obj[_neatKey] = true -- because binser does not detect this private variable

        return obj
    end
end

Genome_new = Genome.new
Genome_newNode = Genome.newNode
Genome_getNode = Genome.getNode
Genome_activate = Genome.activate
Genome_existConnection = Genome.existConnection
Genome_getExcess = Genome.getExcess
Genome_mutateWeight = Genome.mutateWeight
Genome_resetOutputs = Genome.resetOutputs
Genome_newConnection = Genome.newConnection
Genome_getConnection = Genome.getConnection
Genome_newNodes = Genome.newNodes
Genome_mutateNewConnection = Genome.mutateNewConnection
Genome_getAvarageWeightDiff = Genome.getAvarageWeightDiff
Genome_setupMutateProbs = Genome.setupMutateProbs
Genome_mutateChangeActivation = Genome.mutateChangeActivation
Genome_getDisjointConnections = Genome.getDisjointConnections
Genome_mutateSplitConnection = Genome.mutateSplitConnection
Genome_getMatchingConnections = Genome.getMatchingConnections
Genome_existInFreeConnections = Genome.existInFreeConnections
Genome_setAsFreeConnection = Genome.setAsFreeConnection
Genome_setEveryConnectionAsFree = Genome.setEveryConnectionAsFree

math.randomseed(os.time())
random() random() random() random()

return Genome
