// ------------------------------------------------------
// Arduino Push Button Control LED
// LED turns ON when the button is pressed
// LED turns OFF when the button is released
// ------------------------------------------------------
// Define the pin connected to the push button
const int buttonPin = 7;
// Define the pin connected to the LED
const int ledPin = 3;
// Variable to store the current state of the button
int buttonState = 0;
void setup()
{
    // Set LED pin as OUTPUT
    pinMode(ledPin, OUTPUT);
// Set push button pin as INPUT
pinMode(buttonPin, INPUT);
}
void loop()
{
    // Read the current state of the push button
    // It returns HIGH when pressed and LOW when released
    buttonState = digitalRead(buttonPin);
// Check whether the button is pressed
if (buttonState == HIGH)
{
    // Turn the LED ON
    digitalWrite(ledPin, HIGH);

    // You can also write:
    // digitalWrite(ledPin, buttonState);
    // because buttonState is HIGH
}
else
{
    // Turn the LED OFF
    digitalWrite(ledPin, LOW);
}
}