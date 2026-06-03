## Tested under using the same RV for different parents of the same RV
- Correct results 
- No need for optmization of spitting the merged constants at the start 
- The Indexing actually already gets what the vmap demands
- No needed for added complexity
## Added batching constants
- Failing to use 2 constants as None and axis size like in test normal_shared_param
- Used repetitive indexing -> inefficient -> fixed with conditional of repetitive indexing over the correct vmaping axis

## Used rv_equal to further reduce repetitive constants 


## Problem:
Cant use rv_equal on Constant(0) and index([0,1], 0)