# Submittetd Reviews

### R1:
 USER: https://github.com/DonatoLanzillotti/Computational_Intelligence23


Hi Donato, you've been randomly chosen on random.org for my review. I hope you find my comments helpful.

Your methodologies are similar to mine; it seems we're on the right way. I appreciate the idea of creating a new optimal function even though it wasn't required.

I noticed that you evaluate fitness by having an individual play against an opponent whose strategy changes with each move. While this makes results more robust, considering there's a proven optimal strategy for Nim, you could have had them play directly against the "optimal" strategy. This way, the algorithm learns to play against the best opponent and indirectly against all others.

Either way, it seems like you've done a good job. The code comments are helpful; perhaps you could have created a function that directly returns the move based on weights, avoiding if-else conditions in fitness. Anyway, the code is still clear.

Lastly, as a best practice, remember to always include labels in graphs for better readability.

Overall, great work! Feel free to comment on my code if you'd like.


### R2:
USER: https://github.com/rasenqt/computational_intelligence23_24



Hi Michelangelo, I enjoyed your work and want to share some ideas. I'd like to congratulate with you for the README, it was very clear making it easy to understand the code. The game rules you designed are interesting and unconventional, good job. The variation of opponents and shifts when calculating fitness is a clever touch for more robust results. The range of weights is correct, but consider using real numbers from 0 onward to more easily evaluate the convergence of suboptimal strategies. Your implementation is good, but you could introduce a random choice based on the weights for strategy selection and maybe increment the number of games for each individual in order to avoid same fitness results for different individuals. I suggest adding intermediate outputs or a graph to visualize the evolution of the weights. Finally, label the graphs as a best practice for immediate understanding. I hope you find these suggestions useful. Good work!




# Received Reviews

### R1:

USER: https://github.com/rasenqt/computational_intelligence23_24   / Caretto Michelangelo s310178

Hello Nico,
i'm going to review your work , hope you'll enjoy.
I like a lot the way you wrote the code, because is so clean and readable and due to the graph i can easily understand how your "weight" structure works and which strategy is more strong in term of fitness.
I would suggest you , when trying to create an agent playing a game to switch starting player every match too,otherwise your training will be only by one starting side.
Talkink about the Es algortihm , you managed to find a solution in 25 Generation ,due to the fact of "Optimal function" is not optimal , but in general we can not call an algortihm "ES" with this low number of Generation .
Next time maybe with a little curiosity you could study the Nim problem more to make the optimal function stronger.... (less homework done and more curiosity)
I would like to say that despite everything, I appreciated the work and the cleanliness.
Thank you for your effort and attention to detail.
Bye Bye Nico,
Michelangelo.




### R2:

USER: Samuele Vanini s318684 


Overview
The code is well-written and easy to follow. The strategies implemented seem reasonable and are easy to understand.

Areas for Improvement
I have just a couple of considerations about what I feel is not completely right:

All the evolution strategies shown are in the realm of the single-state methods; from what I understood, we also had to implement evolution strategies with a population mu bigger than one.
In "Adaptive (1,λ)-ES" you are forcing the step size of sigma (dividing or multiplying it by 1.5); this is, in principle, wrong. We should not force a certain evolution using fixed parameters but let the algorithm find its way. If you want to balance exploration and exploitation, you can work on the selective pressure (in comma strategy, increase the number of lambda with respect to mu).
Nim is an impartial game with a balanced state if and only if the nim-sum is 0, so you should alternate the player that does the first move during training. Take, for example, the match between 2 expert systems. The first player will always win if the game has an unbalanced initial state, even if the second player has the best strategy. This bias propagates in the training, introducing uncertainty during the fitness evaluation.
Suggestions
The number of generations, 25, is pretty low. The graph shows a fast convergence to 1 for the longest strategy. I would try to find new strategies to balance it.
In "Adaptive (1,λ)-ES" there is a sigma for all the probability; would have been good to see the effect of an adaptive sigma for each weight.
It would have been interesting to see other evolution strategies like (μ + λ) or (μ/ρ + λ)

### R3:
USER: Donato Lanzillotti


PEER REVIEW Nim-Game
POINT 1
The first point consisted of trying the different strategies and understand their performance. Running different games would have allowed you to realizing that the optimal strategy proposed was not totally correct (no 100% winning rate). Althought, changing it was not required.

POINT 2
About the ES, your idea was using the ES in order to find the most appropriate probabilities in choosing the shortest or the longest row. I noticed that you evaluate fitness by counting the number of winning games against a player that uses the optimal strategy. It is reasonable, but playing at the beginning just against a optimal player would not give you useful insight to move, since the winning rate would be always 0. This not the case since the opitmal strategy proposed has not a winning rate equals to 100%.
Thnaks to the graphs it is possible to appreciate the rate at which the probability goes to 0, it means choosing always the longest row.

Overall, it seems you have done a good job.
The code is quite clear to understand but as a suggestions more commments will help into the comprehension.


