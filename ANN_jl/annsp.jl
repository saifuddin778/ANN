module annsp

    values = Dict()
    __init__ = function(data, labels)
                    if annsp.ver__dim(data)
                        values["x"] = data
                        values["y"] = labels
                        values["eta"] = 0.01
                        values["weights"] = randn(length(values["x"][1])+1)
                        annsp.train()
                    else
                        print("no")
                    end
                end

    ver__dim = function(x)
                    random_pick = x[rand(1:length(x))]
                    len_k = 0
                    for a in x
                        len_k = len_k + length(a)
                    end

                    if len_k/length(x) == length(random_pick)
                        return true
                    else
                        return false
                    end
               end

     signum = function(val)
                if val >= 0
                    return 1
                else
                    return -1
                end
              end

     adder = function(w,v)
                return dot(w,v)
             end

     vector_adapt = function(current_weights, x, expected_decision, decision)
                        return current_weights + x*(values["eta"]*(expected_decision-decision))
                    end

     train = function()
                   n_instances = length(values["x"])
                   for i in linspace(1,n_instances, n_instances)
                        insert!(values["x"][i], 1, values["y"][i])
                   end

                   for i in linspace(1,20,20)
                        for j in linspace(1, n_instances, n_instances)
                            expected_decision = values["y"][j]
                            decision = annsp.signum(annsp.adder(values["weights"], values["x"][j]))

                            if decision  != expected_decision
                                values["weights"] = annsp.vector_adapt(values["weights"], values["x"][i], expected_decision, decision)
                            end

                        end
                   end
             end

      get_weights = function()
                        return values["weights"]
                    end

      classify = function(v)
                    j = insert!(v, 1, 1)
                    decision = annsp.signum(annsp.adder(values["weights"], j))
                    return decision
                 end

    return {__init__ = __init__, classify=classify, get_weights = get_weights}
end
