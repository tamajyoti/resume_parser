import click
from algorithm import Model


@click.command()
@click.option("--resume-path",
              is_eager=True,
              required=True,
              help="Absolute path to the resume folder",
              type=click.Path(),
              default="resumes/")
@click.option("--jd-path",
              is_eager=True,
              required=True,
              help="Absolute path to the jd folder",
              type=click.Path(),
              default="resumes/Senior Risk Modelling Analyst")
@click.option("--top-n", default=5, help="top_n_resumes")
def main(resume_path, jd_path, top_n):
    output_dataframe = Model(resume_path, jd_path, top_n).output_dataframe
    print(output_dataframe)


if __name__ == '__main__':
    main()
