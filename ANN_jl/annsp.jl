module annsp
    values = Dict()
    #inits
    __init__ = function(data, labels)
                    ver,len_ = annsp.ver__dim(data)
                    if ver
                        values["x"] = data
                        values["y"] = labels
                        values["eta"] = 0.01
                        values["weights"] = randn(len_+1)
                        annsp.train()
                    else
                        print("no")
                    end
                end

    #verifies data structure and dimensions
    ver__dim = function(x)
                    random_pick = x[rand(1:length(x))]
                    len_k = 0
                    for a in x
                        len_k = len_k + length(a)
                    end
                    if int(len_k/length(x)) == length(random_pick)
                        return true, length(random_pick)
                    else
                        return false
                    end
               end

     #binary signum
     signum = function(val)
                if val >= 0
                    return 1
                else
                    return -1
                end
              end

     #adder function
     adder = function(w,v)
                return dot(w,v)
             end

     #weight vector adaptation method for updating weights
     vector_adapt = function(current_weights, x, expected_decision, decision)
                        return current_weights + x*(values["eta"]*(expected_decision-decision))
                    end

     #train method
     train = function()
                   n_instances = length(values["x"])
                   for i in linspace(1,n_instances, n_instances)
                        insert!(values["x"][int(i)], 1, values["y"][int(i)])
                   end

                   for i in linspace(1,20,20)
                        for j in linspace(1, n_instances, n_instances)
                            expected_decision = values["y"][int(j)]
                            decision = annsp.signum(annsp.adder(values["weights"], values["x"][int(j)]))

                            if decision  != expected_decision
                                values["weights"] = annsp.vector_adapt(values["weights"], values["x"][int(j)], expected_decision, decision)
                            end

                        end
                   end
             end

      #to get the final calculated weights
      get_weights = function()
                        return values["weights"]
                    end

      #simpel classifier
      classify = function(v)
                    j = insert!(v, 1, 1)
                    decision = annsp.signum(annsp.adder(values["weights"], j))
                    return decision
                 end

    return {__init__ = __init__, classify=classify, get_weights = get_weights}
end
