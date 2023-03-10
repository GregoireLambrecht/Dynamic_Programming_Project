#state 1 = Broken 
#state 2 = Old
#state 3 = Good
#state 4 = New

REPAIR

function Maintenance_problem(n::Int64)
    REPAIR::Array{Float64} = [50,30,15]                     #Repartion cost according to the state
    MAINTAIN::Array{Float64} = [1,0.9,0.8]                  #MAINTAIN[j] = proba to stay in j if you maintain 
    NOTHING::Array{Float64} = [1,0.5,0.3]                   #NOTHING[j] = proba to stay in j if you do nothing
    RESULTS::Matrix{Float64} = zeros(4,n)                   #RESULTS[j,i] = expected gain if we are in state j at month i
    STRATEGY = Matrix{String}(undef, 4, n)  #STRATEGY[j,i] = strategy to choose at month i in state j, for the next month
    
    ACTS = ["Replace","Repair","Maintain","Nothing"] 
    #Initialisation
    for i=1:4
        RESULTS[i,n] = (i-1)*10
    end 


    for i = (n-1):-1:1 #Iteration on months
        for j=1:4   #Iteration on states

            if (j==4)   #State = new
                replace = 30 - 70 + RESULTS[j,i+1] #gain if replace
                not = 30 + RESULTS[j-1,i+1]         #gain if nothing

                #choice of the strategy according to not and replace values
                if (replace < not)
                    RESULTS[j,i] = not
                    STRATEGY[j,i] = "Nothing"
                else
                    RESULTS[j,i] = replace
                    STRATEGY[j,i] = "Replace"
                end
            else  
                actions::Array{Float64} = zeros(4)  #actions[1] = if replace, [2] if repair, [3], if maintain, [4] if nothing 
                actions[1] = (j-1)*10 - 70 + RESULTS[4,i+1]         #replace
                actions[2] = (j-1)*10 - REPAIR[j] + RESULTS[3,i+1]  #repair
                if j>1
                    actions[3] = (j-1)*10 - 10 + MAINTAIN[j]*RESULTS[j,i+1] + (1-MAINTAIN[j])*RESULTS[j-1,i+1] #maintain
                    actions[4] = (j-1)*10 + NOTHING[j]*RESULTS[j,i+1] + (1-NOTHING[j])*RESULTS[j-1,i+1]            #nothing
                else
                    actions[3] = (j-1)*10 - 10 + MAINTAIN[j]*RESULTS[j,i+1]
                    actions[4] = (j-1)*10 + NOTHING[j]*RESULTS[j,i+1]
                end   
                act = argmax(actions)
                RESULTS[j,i] = actions[act]
                STRATEGY[j,i] = ACTS[act]
            end
            
        end
    end
    RESULTS,STRATEGY
end

# for i=10:-1:1
#     println(i)
# end
