---
layout: post
title: "Perceptron Visualization"
data: 2015-08-27
categories: jekyll update
---
<head>
  <script type="text/javascript"
          src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
  </script>
</head>

# **Explanation**
  The perceptron algorithm, first introduced by Frank Rosenblatt, is a linear classifier.  
  This means that for a given input $$x \in \mathbb{R}^n$$ the perceptron assigns an output
  label $$y \in \{-1,1\}$$ based on a linear function of the form:\begin{align}
    f(x) &= \text{sgn}(w_0 + w_1x_1 + w_2x_2 +...+ w_nx_n)
  \end{align}
  or written more concisely as:\begin{align}
    f(x) &= \text{sgn}(w^Tx + w_0)
  \end{align}
  This function represents a hyperplane which divides the space into two halves.  This still 
  leaves the question, "How do we learn this hyperplane?"  Well we want to do something like
  (from now on $$w_t$$ will represent $$w$$ at time $$t$$ not at position $$t$$ in the vector):
  
  1. Initialize $$w_0 = \vec{0} \in \mathbb{R}^n$$
  2. For each training example $$(x_i,y_i)$$:
  * Predict $$y' = \text{sgn}(w_t^Tx_i)$$
  * if $$y_i \neq y'$$:
    * Update $$w_{t+1} = w_t + $$something
  
  This is all fine and dandy but how do we update the weights? Well we want to adjust them
  based on how bad the error of the classification was on a given input i.e. $$yx$$.  We can
  read this as if we made an error on input $$x$$ we need to adjust our direction in the direction
  of $$y$$.We also want the learning to happen smoothly.  In other words we don't want the perceptron moving
  wildly everytime it makes an error we want it to take small steps.  Thus we multiply $$yx$$
  by some constant $$\eta$$ usually less than one.  (In the below example $$\eta = .01$$)
  Putting this all together we see that the update rule is: \begin{align}
  w_{t+1} &= w_t + \eta yx
  \end{align}
  
   I've created a visualization to demonstrate how
  the algorithm updates as new examples are seen.  In the below demonstration points are randomly
   generated and assigned a label according to the true function (black line).  Points are
   positive if they are to the right of the line and negative otherwise. 
   
   The  perceptron makes three complete passes over the data before reseting the visualization.  
   The perceptron is learning online each time a new point appears the algorithm re-classifies
   all previously seen points. Every time it makes an error it performs an additive update
   which can be seen when the red line moves.

<!---
+Processing to generate figure
-->
<script type="text/javascript" src="/js/processing.js"></script>
<script type="text/processing" data-processing-target="mycanvas">
        float[] inputs = {12,4};
        float[] weights = {.5,-1};
        Perceptron ptron;
        Trainer[] training = new Trainer[25];
        int count = 0; //element that we are training
        int numPasses = 2; //Number of passes that the perceptron makes over the data
        int passes = 0;
        float alpha = .01;
        float cutoff = 0; 
        float m = 2;
        float b = 1;
        float pm;
        float pb;
        float margin = .25; //The margin around the decision boundary to be considered
        
        
        void setup(){
            size(640,320);
            frameRate(2);
            //initialize Perceptron
            ptron = new Perceptron(3,alpha,cutoff);
            
            // Make 15 initial training points
            for(int i = 0; i < training.length; i++){
                float x = random(-width/2,width/2);
                float y = random(-height/2,height/2);
                int answer = 1;
                if(y < f(x,m,b)){answer = -1;}
                
                training[i] = new Trainer(x,y,answer);
            }
            
            
            
        }
    
    void draw(){
        background(255);
        translate(width/2,height/2);
        //train one point at a time for animation
        ptron.train(training[count].inputs,training[count].answer);
        
        //Once we step through all of the data we do it all over again continuing to traing
        count = (count + 1) % training.length;
        if (count %  training.length == 0) {passes += 1;}
        
        if (passes > numPasses) {
            ptron.weights[0] = 10;
            ptron.weights[1] = 10;
            ptron.weights[2] = 10;
            passes = 0;
        }
        
        stroke(0);
        strokeWeight(3);
        line(width/2,f(width/2,m,b),-width/2,f(-width/2,m,b));
        strokeWeight(1);
        
        //visualize perceptron
        stroke(255,0,0);
        strokeWeight(2);
        pm = -ptron.weights[0]/ptron.weights[1];
        pb = -ptron.weights[2]/ptron.weights[1];
        line(width/2,f(width/2,pm,pb),-width/2,f(-width/2,pm,pb));
        strokeWeight(1);
        
        //visualize direction of positive classification
        //use weights for this
        //line(width/2,f(width/2,pm,pb),-width/2,f(-width/2,pm,pb));
        //strokeWeight(1);
        
        //We visualize the prediction of the perceptron through the most recent training point
        for(int i = 0; i < count; i++){
            stroke(0);
            int guess = ptron.feedforward(training[i].inputs);
            textSize(32);
            fill(0,0,0);
            if(guess > 0) text("-", training[i].inputs[0], training[i].inputs[1]);
            else          text("+", training[i].inputs[0], training[i].inputs[1]);
            
        }
        
    }
    
    
    class Perceptron{
        float[] weights;
        float alpha;
        float cutoff;
        
        Perceptron(int n,float alpha_,float cutoff_){
            weights = new float[n];
            for(int i = 0; i < n; i++){
                weights[i] = 10;//random(-1,1);
            }
            alpha = alpha_;
            cutoff = cutoff_;
            
            
        }
        
        int feedforward(float[] inputs){
            float sum = 0;
            for(int i = 0; i < weights.length; i++){
                sum += inputs[i] * weights[i];
            } 
            return activate(sum);
        }
        
        int activate(float signal){
            if(signal > cutoff) return 1;
            else return -1;
        }
        
        void train(float[] inputs, int label){
            int guess = feedforward(inputs);
            float error = label - guess;
            for(int i = 0; i < weights.length; i++){
                weights[i] += alpha * error * inputs[i]; 
            }
        }
        
        
        
    }
    
    class Trainer{
        int answer;
        float[] inputs;
        Trainer(float x, float y, int a){
            inputs = new float[3];
            inputs[0] = x;
            inputs[1] = y;
            
            inputs[2] = 1;
            answer = a;
        } 
    }
    
    float f(float x,float m, float b){
        return m*x + b; 
    }
    
</script>

<!---
+Buttons for interacting with the sketch
-->

<button onclick="startSketch();">
  Start/Stop
</button>

<!---
<button onclick="stepSketch();">
  Step
</button>


+Javascript used to control the processing sketch
-->

<script type="application/javascript">
        var processingInstance;
        var start = true;
 
         function startSketch() {
           if (start){
              switchSketchState(false);
              start = false
           }else{
              switchSketchState(true);
              start = true;
           }
         }
 
         function stopSketch() {
           switchSketchState(false);
         }
         
         function stepSketch(){
           startSketch();
           stopSketch();
         }
 
         function switchSketchState(on) {
             if (!processingInstance) {
                 processingInstance = Processing.getInstanceById('mycanvas');
             }
 
             if (on) {
                 processingInstance.loop();  // call Processing loop() function
             } else {
                 processingInstance.noLoop(); // stop animation, call noLoop()
             }
         }
         
     </script>
<canvas id="mycanvas"></canvas>

# **Figure Key**
* Black Line -  True decision boundary from which data was labeled
* Red Line - Decision boundary learned by perceptron
* "+" -  Points classified as positive by the perceptron
* "-" - points classified as negative by the perceptron






<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','//www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-68394304-1', 'auto');
  ga('send', 'pageview');

</script>


