from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')

        parser.add_argument('--scene_path', type=str, default='/mnt/data/jywu/dataset/C3VD_origin/SAMPLE_DIR6/cecum_t2_b/images_virtual/')
        parser.add_argument('--query_path', type=str, default='/mnt/data/jywu/code/pytorch-CycleGAN-and-pix2pix/results/cyclegan_c3vd_sampledir6_first_frame/test_60/fake_B')
        parser.add_argument('--checkpoint', type=str, default='/mnt/data/jywu/code/R2Former/CVPR23_DeitS_Rerank.pth', help='path to the checkpoint')
        parser.add_argument('--img_resize', type=int, nargs=2, default=[480, 640], help='image resize')
        parser.add_argument('--save_path', type=str, default='/mnt/data/jywu/code/glace/', help='output file')
        parser.add_argument('--type', type=str, default='target')
        parser.add_argument('--r', type=int, default='4')


        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = True
        return parser
