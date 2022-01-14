
#include <ros.h>
#include <std_msgs/Float32.h>

int fsrPin = 0;        // Vout

ros::NodeHandle  nh;

std_msgs::Float32 vout;
ros::Publisher fsr("fsr", &voltage);

void setup()
{
  nh.initNode();
  nh.advertise(fsr);
}

void loop(void) {
  vout.data = analogRead(fsrPin);
  fsr.publish( &vout );
  nh.spinOnce();
  delay(1000);
}
