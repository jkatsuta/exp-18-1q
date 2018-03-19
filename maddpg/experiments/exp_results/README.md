# Experiment2
In this exp., I focus "simple_tag" and "simple_world_comm".
From the expeiriment1, both max_epi_len of scenarios looked too short to learn.
I changed the max_epi_len here; the results are desribed in 
**my_note/ana2_exp2.ipyhb**.

## exp2-1

#### simple_tag__2018-03-16-_09-49-25
- max_episode_len is 400

#### simple_tag__2018-03-16-_19-15-55
- max_episode_len is 50

#### simple_tag__2018-03-16-_21-44-54
- max_episode_len is 100

--------------------------------------

## exp2-2

#### simple_tag__2018-03-18_00-16-15
- max_episode_len increases by episode (25 -> 200).
    - the episode period where the max_epi_len become twice is 5000 epi.

#### simple_tag__2018-03-18_05-29-24
- max_episode_len increases by episode (25 -> 200).
    - The episode period where the max_epi_len become twice is 10000 epi.
