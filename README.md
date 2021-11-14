DuckHunt
========

Problem Description
-------------------

A description of this exercise is [available on Kattis](https://kth.kattis.com/problems/kth.ai.duckhunt). A screenshot of the problem description is also available in this repository for reference.

Compile
-------

Compile using the Java compiler:
```
javac *.java -d ../bin
```

Modes
-----

The agent can be executed in two different modes:
- *Server* - Act as the judge by sending predefined observations one at a time.
- *Client* - Get observations from std out and output actions to std out (this is the default mode).

Play
----

Run the following command:
```
mkfifo player2server server2player
```

In the first terminal:
```
java Main verbose load XXX.in server < player2server > server2player
```

In the second terminal:
```
java Main verbose > player2server < server2player
```

Or a quickie for debug purposes:
```
java Main server < player2server | java Main verbose > player2server
```

Tips
----

If you receive the error message "getline for player 0 failed", then it is probably because your pipe is broken. Try to remove ``server2player`` and ``player2server`` and run ``mkfifo`` again.

To run the client with assertions enabled, use the ``-ea`` flags.
