#const numDisks = 4.
#const timesteps = 15. % Consider answers of only 15 moves or less

peg(a;b;c).
disk(1..numDisks).
init_on(1..4,a). % Initial state
goal_on(1..4,c). % Goal state

% on(1, 2, 3) means "disk 1 is on peg 2 at timestep 3"

% atop(1, 2) means "disk 1 is on top of disk 2." Allowed if both are disks and the smaller is on top
atop_allowed(Disk, OtherDisk) :- disk(Disk), disk(OtherDisk), Disk < OtherDisk;
