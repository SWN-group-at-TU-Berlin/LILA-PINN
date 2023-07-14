# LILA-PINN
A data-driven leakage detection algorithm enhanced through the estimation of irregular water demands by physics-informed machine learning.

### Underlying work
Daniel et al. (2023) "Estimating irregular water demands with physics-informed machine learning to inform leakage detection" *Water Research* (submitted)

### Authors
[Ivo Daniel](https://www.tu.berlin/swn/ueber-uns/team-1/ivo-daniel), [Andrea Cominola](https://www.tu.berlin/swn/ueber-uns/team-1/andrea-cominola) - [Chair of Smart Water Networks](https://swn.tu-berlin.de) at [Technische Universität Berlin](https://tu.berlin) and [Einstein Center Digital Future, Berlin](https://digital-future.berlin)  

### Organization of repository
- The main results and figures used for manuscript preparation can be found in [publication](publication/).
- All models developed during this work, incl. the physics-informed machine learning algorithm, can be found in [models](models/).
- [_utils](_utils/) contains all data as well as functions used for data loading, helper classes, and the definitions of the functions used for change point detection.

### Dataset
The work in this repository is applied to the dataset of the BattLeDIM.  
(see information below)

### References
Information on the BattLeDIM can be found at:
- https://battledim.ucy.ac.cy/ (Website hosted by comittee)
- https://zenodo.org/record/3902046 (Overview)
- https://zenodo.org/record/4017659#.X4mBaC2w1hE (Dataset)
- https://zenodo.org/record/4139603#.X8lAfbG5p04.mendeley (Competition results)

### LICENSE
Copyright (C) 2021 Ivo Daniel, Simon Letzgus, Andrea Cominola. Released under the [GNU General Public License v3.0](LICENSE). The code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with STREaM. If not, see http://www.gnu.org/licenses/licenses.en.html.
