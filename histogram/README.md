Daniel Pelis
CPE 810 HW 2


Executable can be found in -> ./Debug


The user can utlize the following CLI arguments

-i <VecDim> <BinNum>
-d <GridDim> <BlockDim>

If the 'i' flag is excluded the vec_dim will default to 1000 and the bin_num will default to 64.

If the 'd' flag is excluded the block_dim will default to 32 (size of a warp) and the grid_dim will default to block_dim / vec_dim.