# POSnet
Multimodal POS Tagging
<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- PROJECT LOGO -->
<br />

<h3 align="center">POSNet</h3>

  <p align="center">
    <br />
    <a href="https://github.com/BaraaAlJorf/POSnet"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/BaraaAlJorf/POSnet">View Demo</a>
    ·
    <a href="https://github.com/BaraaAlJorf/POSnet/issues">Report Bug</a>
    ·
    <a href="https://github.com/BaraaAlJorf/POSnet/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
We provide a multimodal POS tagging framework for improved context awareness
<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Installation

Clone the repo onto your machine
   ```sh
   git clone (https://github.com/BaraaAlJorf/POSnet)
   ```
This repository was run and tested with python 3.8 and 3.9, in case of any issues please open contact us by opening the issue on github.
To install dependencies run:
```bash
conda env create --name mynlp --file=environment_droplet.yml
```

### Data processing

to process data run the following script, note that you must have access to (<a href="https://hucvl.github.io/recipeqa/">RecipeQA Dataset</a>). We also provide the processed data in the repository
```bash
python RecipeQA_Dataset.py
```

<!-- USAGE EXAMPLES -->
## Model Train + Eval

To run the training and evaluation scripts, run the following
   ```sh
   python mult_tagger.py
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions or suggestions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

 Baraa Al Jorf - baj321@nyu.edu, Sophie Juco - smj490@nyu.edu, Joel Marvin Tellis - jt4680@nyu.edu, Nirman Taterh - nt2613@nyu.edu

Project Link: https://github.com/BaraaAlJorf/POSnet

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-url]: https://github.com/Zhylkaaa/distributed_causal_discovery/graphs/contributers
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[forks-shield]: https://img.shields.io/github/forks/BaraaaALJorf/Jetbot_Linefollowing.svg?style=for-the-badge
[forks-url]: https://github.com/BaraaAlJorf/Jetbot_Linefollowing//network/members
