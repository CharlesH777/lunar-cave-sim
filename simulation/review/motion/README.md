# 实际数据集预览

每张图为原始 RGB224；面板不裁切、不放大。视频为原始顺序帧五倍速，无插帧。

## seed_20261118_astronaut

> 驶入洞穴内部，遇第一个岔路向右，接近所指的任务目标。洞口只是路标，不是任务终点；真正要靠近并停下的是洞穴深处的宇航员。

[提示词与 CoT 图文](seed_20261118_astronaut/prompt_cot.jpg)

| 家族 | 分区 | query |
|---|---|---|
| train.waypoint_role.statement | train | 驶入洞穴内部，遇第一个岔路向右，接近所指的任务目标。洞口只是路标，不是任务终点；真正要靠近并停下的是洞穴深处的宇航员。 |
| train.waypoint_role.conditional | train | 即使看到了洞口也只是经过路标；要靠近并停车的任务终点是洞穴深处的宇航员。按提示进入洞穴，遇第一个岔路向右，接着走向任务目标。 |
| train.nonterminal_no_stop.statement | train | 不能因为看到了洞口就停；本次要找洞穴深处的宇航员，靠近宇航员后再停下来。进洞行驶，首个分叉走右侧，前往任务目标附近。 |
| train.nonterminal_no_stop.conditional | train | 进洞行驶，在第一分叉转向右边，最后前往任务目标。如果看到洞口，不能把它当作停止理由；要找的是洞穴深处的宇航员，靠近宇航员以后再停。 |
| train.near_goal_only_stop.statement | train | 进洞行驶，首个分叉走右侧，沿着路线接近任务目标。任务目标是洞穴深处的宇航员；只有靠近宇航员才允许停下。 |
| train.near_goal_only_stop.conditional | train | 要找洞穴深处的宇航员，未靠近宇航员时不能停；靠近后才停下来。驶入洞穴内部，于第一分叉选右，继续去任务目标附近。 |
| train.visible_goal_approach.statement | train | 要找的是洞穴深处的宇航员。看到宇航员但尚未靠近时先继续接近，靠近后再停，不能刚看到就停。驶入洞穴内部，于第一分叉选右，再往任务目标所在的位置走。 |
| train.visible_goal_approach.conditional | train | 进洞行驶，遇首处分叉向右转，随后去终点附近。目标是洞穴深处的宇航员；若看见宇航员但还没靠近，就继续接近而非马上停，靠近后再停车。 |
| train.until_near_goal.statement | train | 驶入洞穴内部，到第一个岔口右转，最终前往任务目标。终点是洞穴深处的宇航员附近；在靠近宇航员之前持续前进，直到靠近后再停下。 |
| train.until_near_goal.conditional | train | 只要尚未靠近宇航员就继续前进，直到靠近洞穴深处的宇航员再停车；它才是任务目标。按提示进入洞穴，到第一处分叉选右侧，最后驶向终点。 |
| train.arrival_conjunction.statement | train | 把洞穴深处的宇航员作为任务目标；已按提示进入洞穴与靠近宇航员两个条件都满足后才停。进洞行驶，遇首处分叉向右转，随后前往终点。 |
| train.arrival_conjunction.conditional | train | 沿提示入洞，在第一分叉转向右边，去往任务目标。目标是洞穴深处的宇航员；若已按提示进入洞穴且已靠近宇航员，这两个条件同时成立才可以停下。 |
| train.target_contrast.statement | train | 进洞，在第一个分叉向右转，继续前进至任务目标附近。本次找的是洞穴深处的宇航员，不是洞口；靠近真正的任务目标并停下才结束。 |
| train.target_contrast.conditional | train | 不要把洞口当成要找的目标；本次选择洞穴深处的宇航员作为终点，靠近宇航员才停下来。进洞，遇首处分叉向右转，继续前往任务目标所在位置。 |
| train.route_gate.statement | train | 第一个分叉应走右侧，左侧不是提示方向；真正终点是洞穴深处的宇航员，靠近它才停。驶入洞穴内部，遇首处分叉向右转，继续向任务目标前进。 |
| train.route_gate.conditional | train | 进洞行驶，首个分叉处右转，继续沿路前往任务目标。若来到第一个分叉，就走右边而不是左边；要找的是洞穴深处的宇航员，靠近宇航员之后停止。 |
| heldout.role_and_stop | heldout | 沿提示入洞，首个分叉处右转，接着前往终点附近。虽然洞口只是途经路标而非终点，但只有靠近洞穴深处的宇航员才可以停下来；宇航员才是本次要找的目标。 |
| heldout.nonterminal_and_appearance | heldout | 不能因看到洞口就停；要找的是洞穴深处的宇航员，如果只是看到宇航员但还未靠近，就继续接近，靠近后再停。进洞，首个岔路向右拐，再继续走向终点。 |
| heldout.until_and_preconditions | heldout | 进洞行驶，遇首个岔口往右拐，前往任务目标。任务目标是洞穴深处的宇航员；继续前进，直到已按提示进入洞穴并靠近宇航员这两个条件都满足，再停下来。 |
| heldout.target_and_route_gate | heldout | 要找的是洞穴深处的宇航员而不是洞口；第一个分叉走右不走左，最后靠近真正的任务目标停下。沿提示入洞，遇首个岔口往右拐，最终驶向终点。 |

[整条路线五倍速](seed_20261118_astronaut/preview_5x.mp4)

