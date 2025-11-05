# WORK ON THIS PROJECT HAS MOVED TO [TINY-CHEETAH](https://github.com/risingsunomi/tiny-cheetah)

```
____  _____________________
\   \/  |_____  \__    ___/
 \     / /   |   \|    |   
 /     \/    |    \    |   
/___/\  \_______  /____|   
      \_/       \/       
```

# XOTORCH 
Distributed inference and training with a focus on PyTorch

*Fork of [Exo](https://github.com/exo-explore/exo) v1*

[discord](https://discord.gg/j6bq3E44VR) | [X](https://x.com/shamantekllc)

## Development
Please see the [project task board](https://github.com/orgs/shamantechnology/projects/3) for active tickets.

## Running and Installation
**FOR CUDA USERS: Make sure nvcc is install to detect correct CUDA version for PyTorch install**

If you are not on windows, run the **install.sh** in your terminal

```
$ ./install.sh
```

If you are on Windows, run the **install.ps1** in PowerShell - PowerShell 7.5+ suggested

```
PS C:\> .\install.ps1
```

After install, use the **xot** command

```
$ xot
```

## Background

[[@risingsunomi](https://github.com/risingsunomi)] I first learned about Exo after beginning work on my own distributed inference idea in Ziglang. They appeared on my feed, as I was searching online for resources, via X and I become enthused to help on the project. After working and researching on some of the coding bounties with Exo, I was able to become familiar with the code base. With Exo's focus on [Exo v2](https://x.com/MattBeton/status/1930833977985679362), which appears closed source as of 6/6/2025, and v1 development stopped, I made a decision to continue the work as a hard fork as to not mix this with Exo. I am looking for this project more as more academic pursuit, to solve some early issues and to contribute to open source ecosystem of distributed inference tools.  

The focus on xotorch, and other xo* projects, is more about minimization and focusing on individual Tensor or machine learning libraries instead of all in one. This project is focused on using pytorch and torchtune as, with building out the pytorch inference engine, [PR](https://github.com/exo-explore/exo/pull/139) (not accepted), and talking with some of the community around Exo, pytorch has some better reach on running on different platforms. There are plans for a tinygrad version that will be more built around the library. I hope to bring some novel improvments to this and welcome any PRs or Issues.
