require 'torch'
require 'pl'
--object declaration
local annsp = {}
	
	--inits
	function annsp.init(data, labels)
	    if annsp.verify_dimensions(data) then
	    	annsp.x = data
	    	annsp.y = labels
	    	annsp.eta = 0.01
	    	--annsp.weights = torch.randn(#annsp.x[1]+1)
	    	annsp.weights = torch.Tensor({-0.1839, 0.4486, -0.1335, -0.7360})
	    	annsp.bias = 1
	    	annsp.train()
	    else
	    	print("data not consistent..terminating")
	    end
	end

	--verifies the dimensions and structure of x - if inconsistent, terminate.
	function annsp.verify_dimensions(x)
	    local sum_length = 0
	    local random_pick = #x[torch.random(#x)]
	    for i = 1, #x do
	    	sum_length = sum_length + #x[i]
	    end
	    if sum_length/#x ~= random_pick then
	    	return false
	    else
	    	return true
	    end
	end

	--basic signum (just for demo - can be replaced with either a sigmoid or hyperbolic tangent function)
	function annsp.signum(value)
		if value >= 0 then
			return 1
		else
			return -1
		end
	end

	--adder function
	function annsp.adder(w, x)
		return torch.dot(w,x)
	end

	--adaptation of weight vector - weights updating method
	function annsp.vector_adapt(current_weights, x, expected_decision, decision)
		return current_weights + x*(annsp.eta*(expected_decision-decision))
	end

	--main training function (updates weights on every wrong decision)
	function annsp.train()
		local n_instances = #annsp.x
		--prepending associated class to the vector
		for i=1, n_instances do
			table.insert(annsp.x[i], 1, annsp.y[i])
		end
		annsp.x = torch.Tensor(annsp.x)
		--iterate this many times.
		for l=1, 20 do
			for i=1, n_instances do
				local expected_decision = annsp.y[i]
				local decision = annsp.signum(annsp.adder(annsp.weights, annsp.x[i]))
				if decision ~= expected_decision then
					annsp.weights = annsp.vector_adapt(annsp.weights, annsp.x[i], expected_decision, decision)
					--can add an else condition, which basically will help in terminating the loop over right
					--set of weigts.
				end
			end
		end
	end

	--classifier or predictor function
	function annsp.classify(v)
		local j = v
		table.insert(j, 1, 1)
		j = torch.Tensor(j)
		local decision = annsp.signum(annsp.adder(annsp.weights, j))
		return decision
	end
 
return annsp