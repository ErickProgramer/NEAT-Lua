--- the main file
-- @usage
-- local Neat = require "NEAT.init"
-- @module init

-- @usage
local usage = 'require'

local Neat = {}

--- the @{Genome} module
Neat.Genome = require "NEAT.Genome"

--- the @{Population} module
Neat.Population = require "NEAT.Population"

return Neat
