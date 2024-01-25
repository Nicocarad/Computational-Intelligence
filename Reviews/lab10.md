# Submittetd Reviews

### R1:
 USER: Federico Buccellato s309075

Hi Federico here is my review for your work, I hope you will appreciate it.

First of all, your work seems well done to me and your extension of what we saw in class using Q-learning seems correct.
I have some pointers for you to improve your work and make it better understandable.
I would suggest you to divide your code into classes by perhaps creating a "game" class and a "player" class this allows for better organization and also better reading of the code.
In the training phase you could add an additional "epsilon" parameter to encourage agent exploration by choosing a random action (greedy approach)
The results look good to me, try to increase the number of matches to update the Q-Table you might have better results.
To make the agent more robust, you could do training with different types of players, to be varied randomly during the various iterations just for this reason, creating a player class to be extended could be a good idea.

In conclusion your work seems well done, I only suggest you to improve a bit the organization of your code.
Good work for the next projects!



### R2

USER: Lorenzo Calosso - s306041

Hi Lorenzo here is my review for your work, I hope you will appreciate it.

I really congratulate you as your code is really easy to read and understand due to the use of the classes you created.
The implementation of Q-Learning seems to me to be correct from a theoretical and also implementation point of view.
I don't have many comments to make to you as the code is well written and very similar to my implementation and also the results obtained seem promising.
To further improve your work you could create a "player" class to be extended with different types of players, "random", "minmax", "RL-agent" or other types of players with other strategies, so as to randomly vary the opponent during training making the final agent more robust.
It would also be interesting to see the learning process of your agent as the number of iterations of training changes, so you can figure out the right tradeoff between the number of training iterations and the number of wins.

In conclusion, I again congratulate you on your work.

# Received Reviews

### R1
USER: Paul Raphael

Hello,

Your work is great and very well explained and displayed, the only thing lacking in my opinion is that you don't change the parameters alpha gamma and epsilon during training (maybe you did it on your own or I missed it) It could be interesting.
Aside from that I couldn't find any issues good job and good luck for the exam !


### R2

USER: Angelo Iannielli

Hello Nicolò,

I've conducted a review of your code and wanted to share my observations.

My 2 cents:
Your code is well-structured and organized. I particularly appreciate the clear division into classes, managing both the game logic and player behaviors. This choice significantly enhances the readability and maintainability of the code.

The option to test the trained player in a match against a human player is very interesting. This feature makes your code interactive, providing an immediate test of the achieved results during training.

The integration of graphs at the end of the lab provides a comprehensive view of the learning process. This is a positive touch that offers a visual overview of the agent's performance throughout the training matches.

Recommended Adjustments:
The start_game() function might be considered redundant as it merely prints a static string. You may want to evaluate whether it's essential to keep this function.
The print statements during training could be shortened to enhance file readability. Consider reducing the length of the prints while retaining essential information.
Future Developments:
It could be interesting to explore varying the epsilon variable as the algorithm learns to play. This might help reduce randomness in the agent's moves and reduce exploration during the learning process.
Additionally, it would be compelling to train the agent against players using different strategies. This could provide insights into how well the agent adapts to diverse playing styles.
Overall, you've done an excellent job. Keep it up and consider the suggestions to further enhance your code :)
