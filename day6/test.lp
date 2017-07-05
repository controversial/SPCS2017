% Models airplane flights

bigCity(new_york).
bigCity(san_francisco).
bigCity(denver).
bigCity(miami).
bigCity(seattle).
bigCity(honolulu).
bigCity(jersey_city).
smallCity(poughkeepsie).


flight(new_york, san_francisco).
flight(new_york, denver).
flight(new_york, miami).
flight(miami, seattle).
flight(seattle, honolulu).
flight(san_francisco, honolulu).

train(poughkeepsie, new_york). % If a train goes one way it also goes the opposite way
train(jersey_city, new_york). % If a plane goes one way it also goes the opposite way

% rules

train(B, A) :- train(A, B).
plane(B, A) :- plane(A, B).
city(X) :- bigCity(X).
city(X) :- smallCity(X).
