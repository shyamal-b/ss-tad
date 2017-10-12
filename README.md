# End-to-End, Single-Stream Temporal Action Detection in Untrimmed Videos (SS-TAD)

Welcome to the official repo for "[End-to-End, Single-Stream Temporal Action Detection](https://www.dropbox.com/s/dim9jton0qaw1kj/buch2017bmvc.pdf?dl=0)"! This work was presented as an Oral Talk at [BMVC 2017](https://bmvc2017.london/) in London.

**SS-TAD** is a new, efficient model for generating temporal action _detections_ in untrimmed videos. Analogous to object detections for *images*, temporal action detections provide the temporal bounds in *videos* where actions of interest may lie in addition to their action classes.

<div class="centered">
<a href="http://vision.stanford.edu/pdf/buch2017bmvc.pdf" target="_blank">
<img src="https://dl.dropboxusercontent.com/s/ngl1ofpuwxs1e1b/sstad_modelfig.png" width="590" alt="SS-TAD model overview" />
</div>
<br/>
</a>

This work builds upon our prior work published at [CVPR17](http://cvpr2017.thecvf.com/) on "[SST: Single-Stream Temporal Action Proposals](https://github.com/shyamal-b/sst/)". Now, we are able to provide _end-to-end_ temporal action detection, without requiring a separate classification stage on top of proposals. Furthermore, we observe a _significant_ increase in overall detection performance. For details, please refer to our [paper](https://www.dropbox.com/s/dim9jton0qaw1kj/buch2017bmvc.pdf?dl=0).

### Resources

Quick links:
[[paper](https://www.dropbox.com/s/dim9jton0qaw1kj/buch2017bmvc.pdf?dl=0)]
[[code](https://github.com/shyamal-b/ss-tad/)]
[oral presentation]
<!-- [supplementary] -->

**Note:** Currently, the code in this repo is in *pre-release* - see `code/README.md` for details on planned updates.

Please use the following bibtex to cite our work:

    @inproceedings{sstad_buch_bmvc17,
      author = {Shyamal Buch and Victor Escorcia and Bernard Ghanem and Li Fei-Fei and Juan Carlos Niebles},
      title = {End-to-End, Single-Stream Temporal Action Detection in Untrimmed Videos},
      year = {2017},
      booktitle = {Proceedings of the British Machine Vision Conference ({BMVC})}
      }

If you find this work useful, you may *also* find our prior work of interest:  [SST proposals github repo](https://github.com/shyamal-b/sst/)

<!-- As part of this repo, we also include *evaluation notebooks*, *SS-TAD detections* for THUMOS'14, and *pre-trained model parameters*. Please see the `code/` and `data/` folders for more. -->

<!-- ### Dependencies

We include a *requirements.txt* file that lists all the dependencies you need. Once you have created a [virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/), simply run `pip install -r requirements.txt` from within the environment to install all the dependencies. Note that the original code was executed using Python 2.7. -->

<!-- (For Mac OSX users, you may need to run `pip install --ignore-installed numpy six`) -->
