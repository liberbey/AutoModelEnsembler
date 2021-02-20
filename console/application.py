import logging
import sys

from service.ensemble_classifier_service import EnsembleClassifierService


def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Application started.")
    if len(sys.argv) != 2:
        raise Exception
    input_json_file = sys.argv[1]
    EnsembleClassifierService.run(input_json_file)
    logging.info("Application finished. Please check the output files.")


if __name__ == '__main__':
    main()
