---
layout: post
title: "Perceptron Visualization"
data: 2015-08-27
categories: jekyll update
---
# **Explanation**
  The perceptron algorithm, first introduced by Frank Rosenblatt, is a linear classifier 
  which can be learned in an online fashion.  I've created a visualization to demonstrate how
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

# **Key**
* Black Line -  True decision boundary from which data was labeled
* Red Line - Decision boundary learned by perceptron
* "+" -  Points classified as positive by the perceptron
* "-" - points classified as negative by the perceptron
