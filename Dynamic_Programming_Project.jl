#state 1 = Broken 
#state 2 = Old
#state 3 = Good
#state 4 = New
using Random, Distributions


function Maintenance_problem(n::Int64)
    @time begin
    REPAIR::Array{Float64} = [50,30,15]                     #Repartion cost according to the state
    MAINTAIN::Array{Float64} = [1,0.9,0.8]                  #MAINTAIN[j] = proba to stay in j if you maintain 
    NOTHING::Array{Float64} = [1,0.5,0.3]                   #NOTHING[j] = proba to stay in j if you do nothing
    RESULTS::Matrix{Float64} = zeros(4,n)                   #RESULTS[j,i] = expected gain if we are in state j at month i
    STRATEGY = Matrix{String}(undef, 4, n)                  #STRATEGY[j,i] = strategy to choose at month i in state j, for the next month
    
    ACTS = ["Replace","Repair","Maintain","Nothing"] 
    #Initialisation at month n.
    for i=1:4
        RESULTS[i,n] = (i-1)*10
    end 


    for i = (n-1):-1:1 #Iteration on months
        for j=1:4   #Iteration on states

            if (j==4)   #State = new
                replace = 30 - 70 + RESULTS[j,i+1]         #gain if replace
                not = 30 + RESULTS[j-1,i+1]                #gain if nothing

                #choice of the strategy according to not and replace values
                if (replace < not)
                    RESULTS[j,i] = not
                    STRATEGY[j,i] = "Nothing"
                else
                    RESULTS[j,i] = replace
                    STRATEGY[j,i] = "Replace"
                end
            else  
                actions::Array{Float64} = zeros(4)                  #actions[1] : if replace, [2] if repair, [3] if maintain, [4] if nothing 
                actions[1] = (j-1)*10 - 70 + RESULTS[4,i+1]         #replace
                actions[2] = (j-1)*10 - REPAIR[j] + RESULTS[3,i+1]  #repair
                if j>1
                    actions[3] = (j-1)*10 - 10 + MAINTAIN[j]*RESULTS[j,i+1] + (1-MAINTAIN[j])*RESULTS[j-1,i+1]      #maintain
                    actions[4] = (j-1)*10 + NOTHING[j]*RESULTS[j,i+1] + (1-NOTHING[j])*RESULTS[j-1,i+1]             #nothing
                else
                    actions[3] = (j-1)*10 - 10 + MAINTAIN[j]*RESULTS[j,i+1]
                    actions[4] = (j-1)*10 + NOTHING[j]*RESULTS[j,i+1]
                end   
                act = argmax(actions)                               #Taking the best actions
                RESULTS[j,i] = actions[act]                         
                STRATEGY[j,i] = ACTS[act]
            end
            
        end
    end
    end
    RESULTS,STRATEGY
end


RESULTS, STRATEGY = Maintenance_problem(12)
println(RESULTS)
println(STRATEGY)

println("The answer is :")
print(RESULTS[4,1])

################################################################################################
#EXERCICE 2
################################################################################################ 

p = [0.2, 0.2, 0.4, 0.4, 0.7, 0.7, 0.2, 0.2, 0.8, 0.8, 0.5, 0.5, 0.2, 0.2]
function policySimulator(πt::Matrix{Int64}, n::Int64,T::Int64, draft::Int64) #πt is the policy, n the binomial parameter of the demand, draft the number of simulations for the monte carlo method                                                   
    Bin = x->rand(Binomial(n,x))
    pp = [p[t%14+1] for t=0:(T-1)]  
    R::Float64 = 0                                                  #summing results of the monte carlo method                          
    var::Float64 = 0                                                #The variance
    for k=1:draft                                                   #Iteration for the monte carlo method
        d = Bin.(pp)                                                 #simulation of the demand
        STOCK::Array{Int64} = zeros(T+1)                            #stock
        STOCK[1] = 10
        STOCK[T+1] = 0
        r::Float64 = 0                                                      #The revenue
        for t=1:T
            delivered = d[t] + min(0,STOCK[t] - d[t])                       #number of article delivered
            StockEndDay = STOCK[t]- delivered    #actualisation of the stock
            r += delivered*3 - 0.1 * StockEndDay - πt[STOCK[t]+1,t]          #summing the revenue of the day
            STOCK[t+1] = min(20,StockEndDay + πt[STOCK[t]+1,t])*(t<T+1)
        end
        R+=r
        var+=r*r
    end
    var = var/draft-R*R/(draft*draft)
    size = 2*1.96 * sqrt(var/draft)                                 #size of the confidence interval
    R/draft,size
end


πtest = rand(Binomial(5,0.5),21,14)                                 #πt[j,i] is the number of article bough the day i if STOCK[i] == j

policySimulator(πtest,7,14,10^6)  

function Stock_management(n::Int64,T::Int64)    
    pp = [p[t%14+1] for t=0:(T-1)]                                                           
    REVENUE::Matrix{Float64} = zeros(21,T)                                      #REVENUE[j,t] maximum expect revenue for state j-1 at day t
    STRATEGY::Matrix{Int64} = zeros(21,T)                                       #STRATEGY[j,t] the command for reach the maximum expected revenue at state j-1 the day t
    d = [i for i=0:n]
    deliv::Function = x -> min(0,x)

    #Give the expected value of the next day t+1 when your stock = stock, your command = command, at day t 
    #This is an implementation of calculous made in the last function in a vectorized syntax 
    function evaluateStrat(bin::Binomial{Float64},stock::Int64,command::Int64,t::Int64)::Float64  
        density::Function = x->pdf(bin,x)                                        #The probability density of the day t 
        delivered::Array{Float64} = d .+ deliv.(stock .- d)                      #number of items delivered  
        fsto::Function = x -> min(20,x) 

        StockEndDay::Array{Int64} = stock .- delivered                           #stock at the end of the day (before the command)
        StockNextDay::Array{Int64} = fsto.(StockEndDay .+ command)               #stock the next morning after the command
        rev = [REVENUE[StockNextDay[i]+1,t+1] for i=1:(n+1)]  

        
        if ( t!= T)
            return sum((delivered .* 3 - 0.1 .* StockEndDay .- command .+ rev) .* density.(d))  
        else
            return sum((delivered .* 3) .* density.(d))
        end
    end

    REVENUE[:,T] = [evaluateStrat(Binomial(n,pp[T]),i,0,0) for i=0:20]            #Initialisation at t = T

    for t=(T-1):-1:1                                                              #Iteration on days
        bin = Binomial(n,pp[t])                                                   #The density of the command                                             
        for stock=0:20                                                            #Iteration on states
            eval = x -> evaluateStrat(bin,stock,x,t)                              #returning the expected revenue for command = x
            COMMAND = [i for i=0:5]                                             
            possibleRevenue = eval.(COMMAND)                                      #All the expect revenue for all possible command
            act = argmax(possibleRevenue)                                         #Choising the best command
            REVENUE[stock+1,t] = possibleRevenue[act]                             #Memorizing the maximum expected revenue
            STRATEGY[stock+1,t] = act - 1                                         #Memorizing the command
        end
    end
    REVENUE,STRATEGY
end



#
REVENUE, STRATEGY = Stock_management(7,20)

REVENUE[11,1]
policySimulator(STRATEGY,7,20,10^6)           #REVENUE[11,1] ∈ confidence interval 

