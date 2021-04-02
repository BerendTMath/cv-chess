
#include <math.h>
#include <BasicLinearAlgebra.h>
using namespace BLA;
float *p;

float * set_q(float x,float y,float z, float gripperangle){
    static float q[4];
    float l1 = 30;
    float l2 = 30;
    float l3 = 10;
    
    float q0;
    float q1;
    float q2;
    float q3;

    float base_angle = atan(y/x);
    q0 = base_angle;
    Serial.print("BASE");
    Serial.println(base_angle);

    
    BLA::Matrix<3,1> input_x = {x,y,z};
    BLA::Matrix<3,3> rotate_base = {cos(-base_angle), -sin(-base_angle),0, 
                                    sin(-base_angle), cos(-base_angle),0,
                                    0,0,1};

    BLA::Matrix<3,1> output_x = rotate_base * input_x;

    gripperangle = (gripperangle * PI) / 180;
    
    BLA::Matrix<2,1> p0 = { output_x(0,0) - l3 * cos(gripperangle), z - l3 * sin(gripperangle)};
    BLA::Matrix<1,2> p0T = ~p0;
    

    
    BLA::Matrix<1,1> POT;
    Multiply(p0T, p0, POT);

    q2 = -acos( ( (POT(0,0))  - l1*l1 - l2*l2 ) / (2 * l1 * l2));

    
    q1 = atan( (z - l3 * sin(gripperangle)) / (output_x(0,0) - l3 * cos(gripperangle)) ) - atan( (l2 * sin(q2)) / (l1 + l2 * cos(q2)));
    
    q3 = gripperangle - (q1 + q2);
/*
 *      Serial.print(p0(0,0));
    Serial.println(p0(1,0));
    Serial.println();
 *     
 *  Serial.print("POT: ");
    Serial.print(POT(0,0));
    Serial.println();
    
    Serial.print("Q1: ");
    Serial.print(q1);
    Serial.println();

    Serial.print("Q1: ");
    Serial.print(q2);
    Serial.println();

    Serial.print("Q1: ");
    Serial.print(q3);
    Serial.println();
    
     */
    q[0] = q0 * (180/PI);
    q[1] = q1 * (180/PI);
    q[2] = q2 * (180/PI);
    q[3] = q3 * (180/PI);

    Serial.println();
    Serial.print("base: ");
    Serial.println(q[0]);
    Serial.print("arm1: ");
    Serial.println(q[1]);
    Serial.print("arm2: ");
    Serial.println(q[2]);
    Serial.print("gripper: ");
    Serial.println(q[3]);


//    SET SERVOS
//myservo1.write(q[0], 255);   etc  ...

    return q;
}
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
}
String command;
void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available()){
        command = Serial.readString();
        
        Serial.printf("Command received %s \n", command);

           String h = ",";

       String in[4];
       int count = 0;
       
       for (int i =0; i <= command.length(); i++){
           
           if(command[i] == ',')
           {
              //Serial.print("sdf");
              //Serial.println(command[i]); 
              count++;
           }
           else{
             in[count] += command[i];
             //Serial.println(in[count]);
           }
       }
       Serial.println(in[0] + in[1] + in[2] + in[3]);
       int x1 = in[0].toInt();
       int y1 = in[1].toInt();
       int x2 = in[2].toInt();
       int y2 = in[3].toInt();
    
      Serial.println(x1);
      Serial.println(y1);
      Serial.println(x2);
      Serial.println(y2);

      //SET SERVOS
      p = set_q(37,20,20, 0);
  }
  

  
  delay(100);


}
