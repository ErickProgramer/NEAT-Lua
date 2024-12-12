local Genome = require "Genome"

local g = Genome.new(1,3)
local path= debug.getinfo(1, "S").source
print(path)

