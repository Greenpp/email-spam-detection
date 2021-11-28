from email_spam_detection.preprocess.extract import (
    cleanup,
    create_tmp_dir,
    extract_data,
    merge_data,
)

if __name__ == '__main__':
    print('Creating tmp dir...')
    create_tmp_dir()
    print('Extracting archives...')
    extract_data()
    print('Merging emials data...')
    merge_data()
    print('Cleaning up...')
    cleanup()
    print('Done')
