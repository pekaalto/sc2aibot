###
![](https://media.giphy.com/media/3ov9jITJiJzk9BZUys/giphy.gif)&nbsp;&nbsp;
![](https://media.giphy.com/media/l1J9rVfQSbqAM8rm0/giphy.gif)

![](https://media.giphy.com/media/l1J9sZ8SV7WZjV70s/giphy.gif)&nbsp;&nbsp;
![](https://media.giphy.com/media/3ohhwLxfeO2l0hJa3S/giphy.gif)


### Info

This project implements the FullyConv reinforcement learning agent
for [pysc2](https://github.com/deepmind/pysc2/)
as specified in https://deepmind.com/documents/110/sc2le.pdf.

Differences to the deepmind spec:
- Use [A2C](https://blog.openai.com/baselines-acktr-a2c/) instead of [A3C](https://arxiv.org/abs/1602.01783)
- The non-spatial feature-vector is discarded here.
- There are some other minor simplifaction to the observation space
- Use different hyper-parameters
- And maybe others that I don't know of

### Results
<table align="center">
  <tr>
        <td align="center">Map</td>
        <td align="center">Avg score</td>
        <td align="center">Deepmind avg</td>
    </tr>
    <tr>
        <td align="center">MoveToBeacon</td>
        <td align="center">25</td>
        <td align="center">26</td>
    </tr>
    <tr>
        <td align="center">CollectMineralShards</td>
        <td align="center">91</td>
        <td align="center">103</td>
    </tr>
    <tr>
      <td align="center">DefeatZerglingsAndBanelings</td>
      <td align="center">~training~</td>
      <td align="center">62</td>
    </tr>
    <tr>
      <td align="center">FindAndDefeatZerglings</td>
      <td align="center">42</td>
      <td align="center">45</td>
    </tr>
    <tr>
      <td align="center">DefeatRoaches</td>
      <td align="center">~training~</td>
      <td align="center">100</td>
    </tr>
</table>

- The averages are rough averages.
- For all runs used the default parameters seen in the repo except number of environments.
The scores are from the first run.
- Deepmind scores from the  for FullyConv policy are shown for comparison.
- The model wasn't able to learn CollectMineralsAndGas or BuildMarines

Training graphs:

<p float="left">
<img src="https://image.ibb.co/ih8bT6/Collect_Mineral_Shards.png" width="360" height="300">
<img src="https://preview.ibb.co/cCbfo6/Find_And_Defeat_Zerglings.png" width="360" height="300">
<img src="https://image.ibb.co/cRaZFm/Move_To_Beacon.png" width="360" height="300">
</p>

### How to run
`python run_a2c.py --map_name MoveToBeacon --model_name my_beacon_model --n_envs 32`

This will save
- tf summaries to `_files/summaries/my_beacon_model/`
- model to `_files/models/my_beacon_model`

relative to the project path. See `run_a2c.py` for more arguments.


### Requirements
- Python 3 (will NOT work with python 2)
- [pysc2](https://github.com/deepmind/pysc2/) (tested with v1.2)
- Tensorflow (tested with 1.3.0)
- Other standard python packages like numpy etc.

Code is tested with OS X and Linux. About Windows don't know.
Let me know if there are issues.

### References
I have borrowed some ideas from https://github.com/xhujoy/pysc2-agents (FullyConv-network etc.)
and [Open AI's baselines](https://github.com/openai/baselines/) but the implementation here is different from those.
For parallel environments using the code from baselines adapted for sc2.