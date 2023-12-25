import math

parent_visits = 100
child_visits = 45
prob = 0.8
c2 = 1
c1 = 1
term1 = (math.sqrt(parent_visits) / (1 + child_visits))
term2 = c1 + math.log((parent_visits + c2 + 1) / c2)
partial_utc = prob * term1 * term2
print(partial_utc)
print(f"Term1: {term1}")
print(f"Term2: {term2}")
print(f"Original partial UTC: {prob * term1}")