# ERAv2-S6

Assignment 1

1. From inputs (i) to outputs (a_o):
    1. In the first layer, h1 and h2 are calculated by multiplying weights to the relevant inputs
        1. h1 = (w1 * i1) + (w2 * i2)
        2. h2 = (w3 * i1) + (w4 * i2)
    2. We apply a sigmoid function to h1 and h2 to get a_h1 and a_h2
        1. a_h1 = σ (h1) = 1 / (1 + exp (-h1))
        2. a_h2 = σ (h2) = 1 / (1 + exp (-h2))
    3. In the 2nd layer, o1 and o2 are calculated by multiplying weights to the outputs of the previous layer (a_h1 and a_h2)
        1. o1 = (w5 * a_h1) + (w6 * a_h2)
        2. o2 = (w7 * a_h1) + (w8 * a_h2)
    4. Like #2 above, we apply a sigmoid function to o1 and o2:
        1. a_o1 = σ (o1) = 1 / (1 + exp (-o1))
        2. a_o2 = σ(o2) = 1 / (1 + exp (-o2))
    5. We calculate the error function as a square of the difference between our intended outputs (t1 & t2), and the model’s outputs (a_o1 and a_o2). (We halve the error function to cancel out the 2 when we apply partial derivatives)
        1. E1 = ½ (t1 - a_o1) ^ 2
        2. E2 = ½ (t2 - a_o2) ^ 2
    6. And we calculate our total error function. The goal is to minimise this. 
        1. E = E1 + E2

2. Calculating partial derivatives
    1. We want to calculate the partial derivative of total error for the weight w5. Total error is the sum of E1 and E2. 
        1. ∂E_total / ∂w5 = ∂(E1 + E2) / ∂w5
    2. Since w5 has no effect on E2 (no path to E2), we can leave E2 out. 
        1. ∂E_total / ∂w5 = ∂E1 / ∂w5
    3. We break down the path between w5 and E1 with the intervening elements: o1 and a_o1:
        1. ∂E_total / ∂w5 = ∂E1 / ∂w5 = (∂E1 / ∂a_o1) * (∂a_o1/ ∂o1) * (∂o1 / ∂w5)
    4. And we calculate these individual partial derivatives:
        1. Replace E1 with it’s formula, and apply the derivatives formula
            1. ∂E1 / ∂a_o1 =  ∂ (½ * (t1 - a_o1)²) / ∂a_o1 = 1/2 * 2 * - (t1 - a_o1) = (a_o1 - t1)
        2. Replace a_o1 by σ (a1). And for the sigmoid of x, partial derivative is x * (1 - x):
            1. ∂a_o1 / ∂o1 =  ∂ (σ(o1)) / ∂o1 = a_o1 * (1 - a_o1)
        3. And for the last part, breaking down o1, we know that w6 has no effect (or very little), so is treated as a constant, leaving behind the constant multiplied to w5 (a_h1)
            1. ∂o1 / ∂w5 = ∂((w5 * a_h1) + (w6 * a_h2)) / ∂w5 = a_h1
    5. Combine them to get the derivative for the E_total:
    6. ∂E_total / ∂w5 =  (a_o1 - t1) * a_o1 * (1 - a_o1) * a_h1
        
        
3. Repeat for the other weights:
    1. ∂E_total / ∂w5 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h1
        1. what we calculated
    2. ∂E_total / ∂w6 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h2
        1. Comes from a_h2, so in #1, a_h1 is replaced by a_h2
    3. ∂E_total / ∂w7 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h1
        1. Same as (b), but on the opposite side, so all inverted.
    4. ∂E_total / ∂w8 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h2
        1. Same as (a), with all inverted. 

4. Now to the previous layer, of a_h1 and a_h2. 
    1. Inserting a_o1 (since partial derivative calculated before): 
        1. ∂E1 / ∂a_h1 = (∂E1 / ∂a_o1) * (∂a_o1 / ∂o1) * (∂o1 / ∂a_h1)
    2. And calculating for the three parts:
        1. ∂E1 / ∂a_o1 = (a_o1 - t1)
            1. (from 2.d.i)
        2. ∂a_o1 / ∂o1 = a_o1 * (1 - a_o1)
            1. (from 2.d.ii)
        3. ∂o1 / ∂a_h1 = w5
            1. (since o1 = (w5 * a_h1) + (w6 * a_h2), and derivative w.r.t. a_h1 will be it’s multiplier. 
    3. So for E1, combining the above:
        1.  ∂E1 / ∂a_h1 = (a_o1 - t1) * a_o1 * (1 - a_o1) * w5
    4. And for E2, w.r.t. a_h1, use the lower path with t2, w7, o2, and a_o2:
        1.  ∂E2 / ∂a_h1 = (a_o2 - t2) * a_o2 * (1 - a_o2) * w7
    5. Calculating total error by summing these up:
        1. ∂E_total / ∂a_h1 = ∂(E1 + E2) / ∂a_h1 = 
        (a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7
    6. Calculating the same for a_h2:
        1. ∂E_total / ∂a_h2 = ∂(E1 + E2) / ∂a_h2 =
        (a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8

5. Now moving up 2 layers to the weights, and adding the interstitials h1 and a_h1:
    1. ∂E_total / ∂w1 = (∂E_total / ∂a_h1) * (∂a_h1 / ∂h1) * (∂h1 / ∂w1)
    2. Calculating the 3 parts:
        1. ∂E_total / ∂a_h1 = (a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7
            1. From 4e
        2. ∂a_h1 / ∂h1 = a_h1 * (1 - a_h1)
            1. derivative of sigmoid is x * (1 - x)
        3. ∂h1 / ∂w1 = i1
            1. (input). since h1 = (w1 * i1) + (w2 * i2)
    3. Combining them:
        1. ∂E_total / ∂w1 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1
    4. And calculating for the other weights:
        1. ∂E_total / ∂w2 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2
        2. ∂E_total / ∂w3 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i1
        3. ∂E_total / ∂w4 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i2
        
6. Changing the learning rates:
   See screenshot in base folder. 
