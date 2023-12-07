# Submittetd Reviews

### R1:
 USER: Lorenzo Calosso - s306041

Hi Lorenzo I am sending you my review, I hope you will appreciate it.

In general your work seems to me really well done and structured, I had no difficulties in reading your code thanks to the short texts you inserted before each section. I really appreciated the various comparisons you made using various techniques which led you to choose the best configuration.

Improvements
You used the same number of individuals for both the offsprings and the initial population, this is not a mistake but usually there are more offsprings than population size. You can try changing these parameters to note any reduction in the number of fitness calls while maintaining the same performance.
I recommend you to add a stop condition in case your fitness reaches 1 and also avoid entering a fixed number of generations, the stop criterion you used instead seems very correct to me.
For comparison between the various configurations you used it would also be interesting to have graphs also to show the learning process of your algorithm
Running my code the results obtained in the various instances of the problem were sometimes quite different due to the presence of random elements, since your code contains many random elements I suggest you reevaluate the choices you made by perhaps setting a fixed seed for the random functions.


### R2:
USER:  Michelangelo Caretto s310178

Hi Michelangelo I am writing you this review hoping you will enjoy it.

First of all I thank you for the readme you wrote which allowed me to understand your idea. Next time I suggest you to put markdown comments also before the various code sections so that it will be more understandable.

Your idea of using metrics other than fitness seems interesting and certainly from the results you have shown it allows you to reduce the number of fitness calls while still getting good results. I am not entirely sure that using other metrics besides fitness is "standard" procedure for an EA but after all this seems to work.
The results obtained for the vanilla verison seem a bit low to me, I think the reason is that you imposed a fixed number of generations rather than letting the algorithm go.
You could set an infinite loop that interrupts if the fitness value reaches 1 or if you don't notice any substantial improvement in the last x generations, this certainly increases the number of fitness calls but may also increase the result.
Finally, I suggest you introduce some graphs to show the learning of the algorithm.





# Received Reviews

### R1:

USER: Lorenzo Calosso - s306041

Some considerations:

The code is well written and organized, the markdowns and the comments help to understand what you are doing
The graph and the table at the end are a smart and nice way to present your results
You obtained some good results on the first three instances with respect to the number of fitness calls done
Some advice:

Try to combine also other techniques and see if the fitness improves; personally, I found out that, on this problem, techniques like Elitism, local-search mutation, inversion mutation or also the normal xover, combined together, give an improvement on the results.
For what regard the final fitness of the instance 10, you could try to implement the self adaptive mutation rate instead of using a fixed one: in my case it has given a significant improvement



### R2:

USER: 