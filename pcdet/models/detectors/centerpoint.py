from .detector3d_template import Detector3DTemplate
import numpy as np

class CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            
            points = batch_dict['points'].cpu().detach().numpy()
            bbox_ = batch_dict['final_box_dicts'][0]['pred_boxes'].cpu().detach().numpy()
            np.save("/home/changwon/data/ROS/husky_prediction_bag/visualization_test/{}.npy".format(batch_dict['frame_id'].item().split(".")[0]), points)
            file = open("/home/changwon/data/ROS/husky_prediction_bag/part_a2_result_2/{}.txt".format(batch_dict['frame_id'].item().split(".")[0]), "w")
            with open("/home/changwon/data/ROS/husky_prediction_bag/part_a2_result_2/{}.txt".format(batch_dict['frame_id'].item().split(".")[0]), "w") as f:
                for num in range(bbox_.shape[0]):
                    f.writelines("{},{},{},{},{},{},{},".format(bbox_[num][3],bbox_[num][4],bbox_[num][5],bbox_[num][6],bbox_[num][0],bbox_[num][1],bbox_[num][2]))
            
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
