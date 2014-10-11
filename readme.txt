DuckHunt ai14

A copy of this code is available at GitHub
https://github.com/Realiserad/DuckHunt

A description of this exercise is available at Kattis
https://kth.kattis.com/problems/kth%3Aai%3Aduckhunt

# Compile
javac *.java -d ../bin

# Modes
The agent can be executed in two different modes:
1. Server
Act as the judge by sending predefined observations one at a time.
2. Client
Get observations from std out and output actions to
std out (this is the default mode).

# Play
mkfifo player2server server2player

In first terminal:
java Main verbose load XXX.in server < player2server > server2player

In second terminal:
java Main verbose > player2server < server2player

Or a quickie for debug purposes:
java Main server < player2server | java Main verbose > player2server

# Tips
I receive error message 'getline for player 0 failed'?
This is probably because your pipe is broken, try to remove
server2player and player2server and run mkfifo again.

How can I run the client with assertions on?
Use the -ea flag.
