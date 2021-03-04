# QOXO
A quantum generalization of the game OXO (aka tic-tac-toe).

### compile and run

```bash
make
./qoxo
```

### Rules

#### Notation: 
p: generic player, a: antagonist player;<br>
<img src="https://render.githubusercontent.com/render/math?math=X^{(p)}_i,H^{(p)}_i"> : flip and Hadamard operators applied to the p's symbols on the i site.<br>
<img src="https://render.githubusercontent.com/render/math?math=C^{(p)}_i\_X^{(p[a])}_j"> : CNOT operator with control on site i with p's symbol and target on site j with p's [or a's] symbol;<br> 
if C has many pedices it is understood as a Toffoli gate;<br>
<img src="https://render.githubusercontent.com/render/math?math=\widebar{C}\_X"> denotes an anti-CNOT, that is one acts non-trivially when the controlled qubit is in the state 0.

#### Moves:
Each player can perform certain types of moves, called 'flip', 'bell' and 'mix'. <br>

p) flip \<i\>     &emsp;&emsp;: <img src="https://render.githubusercontent.com/render/math?math=\widebar{C}^{(a)}_i X^{(p)}_i"> <t>[like the usual classical move in OXO];<br>
p) split \<i\> \<j\> : <img src="https://render.githubusercontent.com/render/math?math=\widebar{C}^{(a)}_{ij}\_(X^{(p)}_j H^{(p)}_i C^{(p)}_i\_X^{(p)}_j)">;<br>
p) bell \<i\> \<j\> : <img src="https://render.githubusercontent.com/render/math?math=\widebar{C}^{(a)}_{ij}\_(H^{(p)}_i C^{(p)}_i\_X^{(p)}_j)">;<br>
p) mix \<i\> \<j\>  : <img src="https://render.githubusercontent.com/render/math?math=\widebar{C}^{(a)}_{ij}\_(H^{(p)}_i H^{(p)}_j)">.

#### Winning condition:
After the move of any player p, the 8 winning site triplets with p's symbol (rows, columns and diagonals) are used as control to perform a Toffoli with target on an ancillary qubit (initially set to 0), which is measured for each triple in order to determine whether player p won or not. Notice that after any of these measures the state would collapse on one of the two possible subspaces, the one on which p wins, and the one on which it doesn't (and the antagonist proceeds with the next turn). Therefore, in the latter case, the collapse would cause the winning combinations to disappear!


### Notes
To be honest, there is nothing quantum in this version of the rules, since they can be encoded just in a probabilistic setting. In order to include quantum effects one could introduce, for example, a third parameter encoding a phase, so that interference phenomena between the superposition of the classical games (i.e., in the computational basis) could happen.
