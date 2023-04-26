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

println("#######################################################################################")
println("EXERCIE 1")
println("The Result is :")
println(RESULTS)
println("The Strategy is :")
println(STRATEGY)

println("The answer is :")
print(RESULTS[4,1])

################################################################################################
#EXERCICE 2
################################################################################################ 

println("#######################################################################################")
println("EXERCIE 2")

n = 10

function density(p)
    bin = Binomial(n,p)
    pdf_arr = x -> pdf(bin, float(x))
    pdf_arr.([collect(1:(n+1)) .- 1])
end

p = [0.2, 0.2, 0.4, 0.4, 0.7, 0.7, 0.2, 0.2, 0.8, 0.8, 0.5, 0.5, 0.2, 0.2]

LAW = hcat([[pdf(Binomial(n,p[i]),j) for j=0:n ] for i=1:14]...)

function policySimulator(πt::Matrix{Int64}, n::Int64,T::Int64, draft::Int64) #πt is the policy, n the binomial parameter of the demand, draft the number of simulations for the monte carlo method                                                   
    Bin = x->rand(Binomial(n,x))
    pp = [p[t%14+1] for t=0:(T-1)]  
    R::Float64 = 0                                                  #summing results of the monte carlo method                          
    var::Float64 = 0                                                #The variance
    for k=1:draft                                                   #Iteration for the monte carlo method
        d = Bin.(pp)                                                #simulation of the demand
        STOCK::Array{Int64} = zeros(T+1)                            #stock
        STOCK[1] = 10
        STOCK[T+1] = 0
        r::Float64 = 0                                                          #The revenue
        for t=1:T
            delivered = d[t] + min(0,STOCK[t] - 1 - d[t])                       #number of article delivered
            StockEndDay = STOCK[t]- delivered                                   #actualisation of the stock
            r += delivered*3 - 0.1 * StockEndDay - πt[STOCK[t],t]               #summing the revenue of the day
            STOCK[t+1] = min(20,StockEndDay + πt[STOCK[t],t])*(t<T+1)
        end
        R+=r
        var+=r*r
    end
    var = var/draft-R*R/(draft*draft)
    size = 1.96 * sqrt(var/draft)                                             #confidence interval = [R/draft +- size]
    R/draft,size
end

println("2.d ) A random Strategy :")
πtest = rand(Binomial(5,0.5),20,14)                                             #πt[j,i] is the number of article bough the day i if STOCK[i] == j-1
println(πtest)

println("The result of this strategy is :")
policySimulator(πtest,10,14,10^6)  

function Stock_management(n::Int64,T::Int64)    
    pp = [p[t%14+1] for t=0:(T-1)]  
    LAW = hcat([[pdf(Binomial(n,pp[i]),j) for j=0:n ] for i=1:T]...)                                                      
    REVENUE::Matrix{Float64} = zeros(20,T+1)                                      #REVENUE[j,t] maximum expect revenue for state j-1 at day t
    STRATEGY::Matrix{Int64} = zeros(20,T)                                       #STRATEGY[j,t] the command for reach the maximum expected revenue at state j-1 the day t
    d = [i for i=0:n]
    part_neg::Function = x -> min(0,x)                                         

    #Give the expected value of the next day t+1 when your stock = stock, your command = command, at day t 
    #This is an implementation of calculous made in the last function in a vectorized syntax 
    function evaluateStrat(stock::Int64,command::Int64,t::Int64)::Float64   
        delivered::Array{Float64} = d .+ part_neg.(stock .- 1 .- d )             #number of items delivered  
        fsto::Function = x -> min(20,x) 
        StockEndDay::Array{Int64} = stock .- delivered                           #stock at the end of the day (before the command)
        StockNextDay::Array{Int64} = fsto.(StockEndDay .+ command)               #stock the next morning after the command
        rev = [REVENUE[StockNextDay[i],t+1] for i=1:(n+1)]  

        #doesn't need to check if t = T 
        sum((delivered .* 3 - 0.1 .* StockEndDay .- command .+ rev) .* LAW[1:n+1,t])  
    end

    for stock=1:20
        REVENUE[stock,T] = evaluateStrat(stock,0,T)                               #Initialisation at t = T
    end

    for t=(T-1):-1:1                                                              #Iteration on days                                         
        for stock=1:20                                                            #Iteration on states
            eval = x -> evaluateStrat(stock,x,t)                                  #returning the expected revenue for command = x
            COMMAND = [i for i=0:5]                                             
            possibleRevenue = eval.(COMMAND)                                      #All the expect revenue for all possible command
            act = argmax(possibleRevenue)                                         #Choising the best command
            REVENUE[stock,t] = possibleRevenue[act]                               #Memorizing the maximum expected revenue
            STRATEGY[stock,t] = act - 1                                           #Memorizing the command
        end
    end
    REVENUE,STRATEGY
end

println("2.e )For T = 14 the answer is :")
REVENUE, STRATEGY = Stock_management(10,14)

println("The Strategy :")
println(STRATEGY)
println("The revenue :")
println(REVENUE)
println("The maximum expected income :")
println(REVENUE[10,1])
println("2.f )With Monte carlo : (expected value, size of the interval)")
policySimulator(STRATEGY,10,14,10^6)           #REVENUE[11,1] ∈ confidence interval  


println("3.a )For T = 96 the answer is :")
REVENUE, STRATEGY = Stock_management(10,96)
println(STRATEGY)
println("The revenue :")
println(REVENUE)
println("The maximum expected income :")
println(REVENUE[10,1])
println("With Monte carlo : (expected value, size of the interval)")
policySimulator(STRATEGY,10,96,10^4)           #REVENUE[11,1] ∈ confidence interval  


function Stock_management_2(n::Int64,T::Int64)    
    pp = [p[t%14+1] for t=0:(T-1)]  
    LAW = hcat([[pdf(Binomial(n,p[i]),j) for j=0:n ] for i=1:T]...)                                                         
    REVENUE::Array{Float64} = zeros(20,6,T+1)                                      #REVENUE[j,t] maximum expect revenue for state j-1 at day t
    STRATEGY::Array{Int64} = zeros(20,6,T)                                         #STRATEGY[j,t] the command for reach the maximum expected revenue at state j-1 the day t
    d = [i for i=0:n]
    part_neg::Function = x -> min(0,x)

    #Give the expected value of when your stock = stock, your command = command, at day t 
    #This is an implementation of calculous made in the last function in a vectorized syntax 

    function evaluateStrat(stock::Int64,inComing::Int64,command::Int64,t::Int64)::Float64
        d = [i for i=0:n] 
        delivered::Array{Float64} = d .+ part_neg.(stock .- 1 .- d)               #number of items delivered  
        fsto::Function = x -> min(20,x) 
    
        StockEndDay::Array{Int64} = stock .- delivered                           #stock at the end of the day (before the command)
        StockNextDay::Array{Int64} = fsto.(StockEndDay .+ inComing)              #stock the next morning after the command
        rev = [REVENUE[StockNextDay[i],command+1,t+1] for i=1:(n+1)]             #futures revenues
        sum((delivered .* 3 - 0.1 .* StockEndDay .- inComing .+ rev) .* LAW[1:n+1,t]) 
    end

    for stock=1:20
        for command=0:5
            REVENUE[stock,command+1,T] = evaluateStrat(stock,command,0,T)         #Initialisation at t = T
        end
    end

    for t=(T-1):-1:1                                                              #Iteration on days                                         
        for stock=1:20                                                            #Iteration on stocks
            for inComing=0:5
                eval = x -> evaluateStrat(stock,inComing,x,t)                     #returning the expected revenue for command = x
                COMMAND = [i for i=0:5]                                             
                possibleRevenue = eval.(COMMAND)                                  #All the expect revenue for all possible command
                act = argmax(possibleRevenue)                                     #Choising the best command
                REVENUE[stock,inComing+1,t] = possibleRevenue[act]                #Memorizing the maximum expected revenue
                STRATEGY[stock,inComing+1,t] = act - 1                            #Memorizing the command
            end
        end
    end
    REVENUE,STRATEGY
end

println("3.b ) For T = 14 the answer is :")
REVENUE,STRATEGY = Stock_management_2(10,14)

println("The Strategy :")
println(STRATEGY)
println("The revenue :")
println(REVENUE)
println("The maximum expected income :")
println(REVENUE[10,1,1])



################################################################################################
#EXERCICE 3
################################################################################################ 
println("#######################################################################################")
println("EXERCIE 3")

#Naive Strat : Always Buy dice when it possible
#Return 0 if we don't buy
#1 else
function naiveStrat(t::Int64, points::Int64, dices::Int64)::Int64
    if (points >= 6) && (dices < 3)
        return 1
    end
    0
end

function policySimulator(Strat::Function, T::Int64, draft::Int64)   #draft the number of simulations for the monte carlo method                                                    
    R::Float64 = 0                                                  #summing results of the monte carlo method                          
    var::Float64 = 0                                                #The variance
    for k=1:draft                                                   #Iteration for the monte carlo method
        points::Int64 = 0
        dices::Int64 = 1
        for t=1:T
            act::Int64 = Strat(t, points, dices)                    #Get the action according to the state
            dices += act                                            #actualisation of the number of dices 
            results = maximum(rand(DiscreteUniform(1,6),dices))     #Cast dices and get the maximum result 
            points += results - 5*act                               #actualisation of points
        end
        R+=points
        var+=points*points
    end
    var = var/draft-R*R/(draft*draft)
    size = 1.96 * sqrt(var/draft)                                             #confidence interval = [R/draft +- size/2]
    R/draft,size
end

println("The naive strategy : Buy a dice when it is possible")
println("Result with policy simulator :")
policySimulator(naiveStrat, 10,1000)

function max3(k::Int64)
    (3*k*(k-1) + 1) * (1/6)^3
end

function law(nDices::Int64,k::Int64)
    (k/6)^nDices - ((k-1)/6)^nDices
end


#The LAW of max(X1), max(X1,X2), max(X1,X2,X3)
#LAW[i,k] = P(max(X1,...,Xi) = k)

LAW = transpose(hcat([(x->law(nDices,x)).(collect(1:6)) for nDices=1:10]...))

#T is the time
#nDices is the maximum number of dices 
function Simple_game(T::Int64,nDices::Int64)
    @time begin
    nPoints::Int64 = T*6+1
    RESULTS::Array{Float64} = zeros(nDices,nPoints, T)               #RESUTLS[i,j,t] = expected points if we have j points and i dices at day t  
    STRATEGY::Array{Int64} = zeros(nDices, nPoints, T)               #STRATEGY[i,j,t] = strategy to choose at day t if j points and i dices

    for i=1:nDices
        for j=1:nPoints
            RESULTS[i,j,T] = j-1                                     #Initialisation at day T
        end
    end

    for t=(T-1):-1:1                                                 #Iteration on times
        for i=1:nDices                                               #Iteration on dices
            for j=1:(nPoints - (T-t)*6)                              #Iteration on points 
                if i==nDices                                                                                #Impossible to buy more dices
                    RESULTS[i,j,t] = sum(LAW[i,:] .* RESULTS[i,(j+1):(j+6),t+1])
                else
                    if (j-1<=5)                                                                             #Impossible to buy more dices
                        RESULTS[i,j,t] = sum(LAW[i,:] .* RESULTS[i,(j+1):(j+6),t+1])
                    else                                                                                    #Possible to buy more dices
                        Buy = sum( LAW[i+1,:] .* RESULTS[i+1,(j+1):(j+6),t+1]) -5                           #Result if you buy a dice                        
                        dontBuy = sum(LAW[i,:] .* RESULTS[i,(j+1):(j+6),t+1])                               #If you don't buy
                        STRATEGY[i,j,t] = Buy >= dontBuy                                                    #Do the best moove
                        RESULTS[i,j,t] = max(Buy, dontBuy)                                                  #Get the best result
                    end
                end
            end
        end
    end
    end
    RESULTS, STRATEGY
end


RESULTS, STRATEGY = Simple_game(11,3)
println("4.e )")
println("Results:")
println(RESULTS)
println("STRATEGY:")
println(STRATEGY)
println("Maximum expected points : ")
println(RESULTS[1,1,1])


#Best Strategy as a function
function πstar(t::Int64, points::Int64, dices::Int64)::Int64
    STRATEGY[dices,points+1,t]
end  

println("4.f )")
println("RESULTS with Monte Carlo :")
policySimulator(πstar,10,100000)


#T is the time
#nDices is the maximum number of dices 
function Simple_game_2(T::Int64,nDices::Int64,SELLING)
    @time begin
    nPoints::Int64 = T*12 
    RESULTS::Array{Float64} = zeros(nDices,nPoints, T)               #RESUTLS[i,j,t] = expected points if we have j points and i dices at day t  
    STRATEGY::Array{Int64} = zeros(nDices, nPoints, T)               #STRATEGY[i,j,t] = strategy to choose at day t if j points and i dices
    
    K = [0,2,4,5,8]

    for i=1:nDices
        for j=1:nPoints
            RESULTS[i,j,T] = j-1 + K[i]*SELLING                                     #Initialisation at day T
        end
    end

    for t=(T-1):-1:1                                                 #Iteration on times
        for i=1:nDices                                               #Iteration on dices
            for j=1:(nPoints - (T-t)*12)                              #Iteration on points 
                if i==nDices                                                                                #Impossible to buy more dices
                    #RESULTS[i,j,t] = sum(LAW[i,:] .* RESULTS[i,(j+1):(j+6),t+1])
                    dontSplit = sum(LAW[i,:] .* RESULTS[i,(j+1):(j+6),t+1])
                    Split = sum(LAW[i,:] .* RESULTS[i-1,(j+2):2:(j+12),t+1])
                    STRATEGY[i,j,t] = 2 * (Split > dontSplit)
                    RESULTS[i,j,t] = max(Split,dontSplit)
                else
                    
                    if (j-1<=5)                                                                             #Impossible to buy more dices
                        if (i==1)
                            RESULTS[i,j,t] = sum(LAW[i,:] .* RESULTS[i,(j+1):(j+6),t+1])
                        else 
                            dontSplit = sum(LAW[i,:] .* RESULTS[i,(j+1):(j+6),t+1])
                            Split = sum(LAW[i,:] .* RESULTS[i-1,(j+2):2:(j+12),t+1]) 
                            STRATEGY[i,j,t] = 2 * (Split > dontSplit)
                            RESULTS[i,j,t] = max(Split,dontSplit)
                        end

                    else
                        if (i==1)                                                                           #Possible to buy more dices
                            Buy = sum( LAW[i+1,:] .* RESULTS[i+1,(j+1):(j+6),t+1]) -5                           #Result if you buy a dice                        
                            dontBuy = sum(LAW[i,:] .* RESULTS[i,(j+1):(j+6),t+1])                               #If you don't buy
                            STRATEGY[i,j,t] = Buy >= dontBuy                                                    #Do the best moove
                            RESULTS[i,j,t] = max(Buy, dontBuy)                                                  #Get the best result
                        else 
                            Buy_dontSplit = sum( LAW[i+1,:] .* RESULTS[i+1,(j+1):(j+6),t+1]) -5                           #Result if you buy a dice                        
                            dontBuy_dontSplit = sum(LAW[i,:] .* RESULTS[i,(j+1):(j+6),t+1])                               #If you don't buy
                            Buy_Split = sum( LAW[i+1,:] .* RESULTS[i,(j+2):2:(j+12),t+1]) -5
                            dontBuy_Split = sum(LAW[i,:] .* RESULTS[i-1,(j+2):2:(j+12),t+1]) 
                            STRATEGY[i,j,t] = argmax([dontBuy_dontSplit,Buy_dontSplit,dontBuy_Split,Buy_Split])-1                                                  #Do the best moove
                            RESULTS[i,j,t] = max([dontBuy_dontSplit,Buy_dontSplit,dontBuy_Split,Buy_Split]...) 
                        end
                    end
                end
            end
        end
    end
    end
    RESULTS, STRATEGY
end


RESULTS1, STRATEGY1 = Simple_game_2(11,5, false)
RESULTS2, STRATEGY2 = Simple_game_2(1000,5, false)

