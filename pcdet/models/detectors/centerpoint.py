from .detector3d_template import Detector3DTemplate
import numpy as np # for GT box visualization

class CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        #if batch_dict['frame_id'].item().split(".")[0] == "merged_1622684874_2832005024":
        # if True:
        #     gt_=[]
        #     #for GT box visualization in forward 
        #     # lwh, heading, xyz <= st3d
        #     # where, xyz,lwh,heading
        #     gt_box = batch_dict['gt_boxes'].squeeze(0).cpu().detach().numpy()
        #     #gt_box = [[gt_box[0][3], gt_box[0][4], gt_box[0][5]], gt_box[0][6], [gt_box[0][0], gt_box[0][1], gt_box[0][2]]] 
        #     points = batch_dict['points'].cpu().detach().numpy()
        #     np.save("/home/changwon/detection_task/SSOD/kakao/SSDA/visualization_in_model/{}.npy".format(batch_dict['frame_id'].item().split(".")[0]), points)
        #     file = open("/home/changwon/detection_task/SSOD/kakao/SSDA/visualization_in_model/{}.txt".format(batch_dict['frame_id'].item().split(".")[0]), "w")
        #     with open("/home/changwon/detection_task/SSOD/kakao/SSDA/visualization_in_model/{}.txt".format(batch_dict['frame_id'].item().split(".")[0]), "w") as f:
        #         for num in range(gt_box.shape[0]):
        #             f.writelines("{},{},{},{},{},{},{},".format(gt_box[num][3],gt_box[num][4],gt_box[num][5],gt_box[num][6],gt_box[num][0],gt_box[num][1],gt_box[num][2]))
        #             gt_.append([gt_box[num][3],gt_box[num][4],gt_box[num][5],gt_box[num][6],gt_box[num][0],gt_box[num][1],gt_box[num][2]])
        #     #scene_viz(gt_box, points)
        #     token = batch_dict['metadata'][0]['token']
        #     print(batch_dict['frame_id'].item().split(".")[0])

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts, _ = self.post_processing(batch_dict)
            # points = batch_dict['points'].cpu().detach().numpy()
            # np.save("/home/changwon/detection_task/SSOD/kakao/SSDA/visualization_in_model/{}.npy".format(batch_dict['frame_id'].item().split(".")[0]), points)
            # file = open("/home/changwon/detection_task/SSOD/kakao/SSDA/visualization_in_model/pred_box.txt", "w")
            # with open("/home/changwon/detection_task/SSOD/kakao/SSDA/visualization_in_model/pred_box.txt", "w") as f:
            #     for num in range(pred_dicts[0]['pred_boxes'].shape[0]):
            #         f.writelines("{},{},{},{},{},{},{},".format(pred_dicts[0]['pred_boxes'][num][3],pred_dicts[0]['pred_boxes'][num][4],pred_dicts[0]['pred_boxes'][num][5],pred_dicts[0]['pred_boxes'][num][6],pred_dicts[0]['pred_boxes'][num][0],pred_dicts[0]['pred_boxes'][num][1],pred_dicts[0]['pred_boxes'][num][2]))
            return pred_dicts, recall_dicts, _

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

        return final_pred_dict, recall_dict, None