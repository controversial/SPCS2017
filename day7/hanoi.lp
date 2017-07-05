#const numDisks = 4.
#const timesteps = 15. % Consider answers of only 15 moves or less

peg(a;b;c). % Define the pegs called a, b, and c
disk(1..numDisks). % Define disks called 1 to 4
init_on(1..4,a). % Begin with all disks on a
goal_on(1..4,c). % Goal is all disks on c

end(timesteps). % The end is timestep 15

% TERMINOLOGY
% on(1, a, 3) means "disk 1 is on peg a at timestep 3"
% move(1, c, 3) means "disk 1 is moved to peg c at timestep 3"


% For every T from 1 to the number of allowed timesteps, choose ONE move with Disk as a disk and Peg as a peg
{ move(Disk, Peg, Time) : disk(Disk), peg(Peg) }=1 :- Time=1..timesteps.

% If a given disk is moved to a given peg at a given time, that disk must be ON that peg at that time.
% If a disk is moved, then it's on its destination peg.
on(Disk, Peg, Time) :- move(Disk, Peg, Time).
% It could also be there from the start, in which case there would not be a move for it
on(Disk, Peg, Time) :- init_on(Disk, Peg, Time).
% If a disk wasn't moved, it will still be there at the next timestep
on(Disk, Peg, Time + 1) :- on(Disk, Peg, Time),
                           not move(Disk, _, Time + 1), % No subsequent move (to any peg)
                           not end(Time).               % Not out of time

% We say a disk is "blocked" if it's underneath another disk.
% We can tell a disk is under another disk if there's a smaller disk on the same peg

% A disk is blocked if...
blocked(Under, Time) :- on(Over, Peg, Time), on(Under, Peg, Time), % Two disks are on the same peg, which we'll call under and over. Under is the one we're looking at.
                        disk(Over; Under), % Both are disks
                        Under<Over.        % The disk is larger than the other
% Therefore, a disk will be marked as "blocked" if it is a disk and there is ever another smaller disk on the peg

% A move is "unstable" if it involves placing a larger disk on top of a smaller disk.
% Over is the disk we're moving. It will always be over any other disks because it is being placed on top.
unstable(move(Over, Peg, Time)) :- on(Under, Peg, Time-1), % There's another disk which we'll call "under" on this peg at the previous time
                                   move(Over, Peg, Time),
                                   disk(Over;Under),       % Both are disks
                                   Over < Under.           % The other disk is smaller


% Now we constrain the world to prevent making unstable or blocked moves. We eliminate worlds where...

% There are blocked moves
:- move(Disk, Peg, Time), blocked(Disk, Time-1). % False if there's a move and the disk is blocked
% There are unstable moves
:- move(Disk, Peg, Time), unstable(move(Disk, Peg, Time)). % False if there is a move and it's unstable
% The disks don't appear in the end state
:- goal_on(Disk, Peg), not on(Disk, Peg, Time), end(Time). % False if it's the end time and the state of a disk doesn't match the goal state of a disk
% No disks appear in multiple places at any time
:- { on(Disk, Peg, Time) } != 1, disk(Disk), end(Time), Time = 1..timesteps.

% atop(1, 2) means "disk 1 is on top of disk 2." Allowed if both are disks and the smaller is on top
atop_allowed(Disk, OtherDisk) :- disk(Disk), disk(OtherDisk), Disk < OtherDisk.
